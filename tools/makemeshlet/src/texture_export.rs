//! 语义感知 PBR 贴图导出与 ORM 通道打包模块
//!
//! 解决直接盲目缩放导出 PNG 带来的画质与通道错位问题：
//! 1. **Albedo / Emissive (sRGB)**：先解码至线性空间做面积加权滤波降采样，再回写 sRGB，避免色彩发暗与色偏；
//! 2. **Normal Map**：支持 2 通道 (`R8G8`/`R16G16`) 自动重构 Z 分量；降采样时在 `[-1, 1]` 向量空间滤波后逐像素重归一化，保持法线单位长度与微表面锐度；
//! 3. **ORM (Occlusion + Roughness + Metallic)**：
//!    - 自动遵循 glTF 标准布局（`R = Occlusion`, `G = Roughness`, `B = Metallic`）；
//!    - 兼容已合并 ORM 贴图、分离的 AO + MR 贴图（自动重采样合并为单张 ORM，使运行时仅加载 1 张贴图）、以及 2 通道 `R8G8`/`R16G16` 粗糙度/金属度贴图（映射至 G/B 避免金属度清零）。

use std::collections::HashMap;
use std::path::Path;

use gltf::image::Format;

/// 贴图导出器：按语义角色去重缓存，避免同源贴图重复编码落盘。
pub struct SemanticTextureExporter<'a> {
    images: &'a [gltf::image::Data],
    out_dir: &'a Path,
    stem: &'a str,
    max_texture_size: u32,
    pack_orm: bool,
    enabled: bool,
    dir_created: bool,
    srgb_cache: HashMap<usize, Option<String>>,
    normal_cache: HashMap<usize, Option<String>>,
    orm_cache: HashMap<(Option<usize>, Option<usize>), (Option<String>, Option<String>)>,
    pub exported_count: usize,
    pub resized_count: usize,
    pub orm_packed_count: usize,
}

impl<'a> SemanticTextureExporter<'a> {
    pub fn new(
        images: &'a [gltf::image::Data],
        out_dir: &'a Path,
        stem: &'a str,
        max_texture_size: u32,
        pack_orm: bool,
        enabled: bool,
    ) -> Self {
        Self {
            images,
            out_dir,
            stem,
            max_texture_size,
            pack_orm,
            enabled,
            dir_created: false,
            srgb_cache: HashMap::new(),
            normal_cache: HashMap::new(),
            orm_cache: HashMap::new(),
            exported_count: 0,
            resized_count: 0,
            orm_packed_count: 0,
        }
    }

    fn ensure_textures_dir(&mut self) -> Result<(), std::io::Error> {
        if !self.dir_created {
            std::fs::create_dir_all(self.out_dir.join("textures"))?;
            self.dir_created = true;
        }
        Ok(())
    }

    /// 导出 sRGB 颜色贴图（BaseColor / Emissive），降采样时自动在 Linear 空间滤波。
    pub fn export_srgb(&mut self, img_idx: Option<usize>) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let idx = img_idx?;
        if let Some(cached) = self.srgb_cache.get(&idx) {
            return cached.clone();
        }
        let Some(img_data) = self.images.get(idx) else {
            self.srgb_cache.insert(idx, None);
            return None;
        };
        if let Err(err) = self.ensure_textures_dir() {
            eprintln!("  [警告] 创建 textures 目录失败: {err}");
            return None;
        }

        let rel_path = format!("textures/{}_tex_{}.png", self.stem, idx);
        let abs_path = self.out_dir.join(&rel_path);
        let result = match save_srgb_color_png(img_data, &abs_path, self.max_texture_size) {
            Ok(was_resized) => {
                self.exported_count += 1;
                if was_resized {
                    self.resized_count += 1;
                }
                Some(rel_path)
            }
            Err(err) => {
                eprintln!("  [警告] sRGB 贴图 #{idx} 导出失败，已跳过: {err}");
                None
            }
        };
        self.srgb_cache.insert(idx, result.clone());
        result
    }

    /// 导出切线空间法线贴图（支持 2 通道 Z 重构 + 降采样后逐像素重归一化）。
    pub fn export_normal(&mut self, img_idx: Option<usize>) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let idx = img_idx?;
        if let Some(cached) = self.normal_cache.get(&idx) {
            return cached.clone();
        }
        let Some(img_data) = self.images.get(idx) else {
            self.normal_cache.insert(idx, None);
            return None;
        };
        if let Err(err) = self.ensure_textures_dir() {
            eprintln!("  [警告] 创建 textures 目录失败: {err}");
            return None;
        }

        let rel_path = format!("textures/{}_normal_{}.png", self.stem, idx);
        let abs_path = self.out_dir.join(&rel_path);
        let result = match save_normal_png(img_data, &abs_path, self.max_texture_size) {
            Ok(was_resized) => {
                self.exported_count += 1;
                if was_resized {
                    self.resized_count += 1;
                }
                Some(rel_path)
            }
            Err(err) => {
                eprintln!("  [警告] 法线贴图 #{idx} 导出失败，已跳过: {err}");
                None
            }
        };
        self.normal_cache.insert(idx, result.clone());
        result
    }

    /// 导出 Occlusion 与 MetallicRoughness 贴图：
    /// - 当两者同源，或开启 `pack_orm` 且两者均存在时，合并导出为单张标准 3 通道 ORM（R=O, G=R, B=M），
    ///   返回相同的相对路径以保证运行时 `AssetServer` 仅占一份显存；
    /// - 当仅存在 MR（含 2 通道 `R8G8`/`R16G16`）时，规范化为 `[1.0, R, M]` RGB8 PNG，防止 2 通道被错填 0 导致金属度丢失。
    pub fn export_orm_pair(
        &mut self,
        occ_idx: Option<usize>,
        mr_idx: Option<usize>,
    ) -> (Option<String>, Option<String>) {
        if !self.enabled || (occ_idx.is_none() && mr_idx.is_none()) {
            return (None, None);
        }

        let cache_key = (occ_idx, mr_idx);
        if let Some(cached) = self.orm_cache.get(&cache_key) {
            return cached.clone();
        }

        if let Err(err) = self.ensure_textures_dir() {
            eprintln!("  [警告] 创建 textures 目录失败: {err}");
            return (None, None);
        }

        let result = match (occ_idx, mr_idx) {
            // 1. glTF 中本就是同一张 ORM 贴图
            (Some(o), Some(m)) if o == m => {
                if let Some(img) = self.images.get(o) {
                    let rel_path = format!("textures/{}_orm_{}.png", self.stem, o);
                    let abs_path = self.out_dir.join(&rel_path);
                    match save_orm_png(Some(img), Some(img), &abs_path, self.max_texture_size) {
                        Ok(was_resized) => {
                            self.exported_count += 1;
                            if was_resized {
                                self.resized_count += 1;
                            }
                            (Some(rel_path.clone()), Some(rel_path))
                        }
                        Err(err) => {
                            eprintln!("  [警告] ORM 贴图 #{o} 导出失败，已跳过: {err}");
                            (None, None)
                        }
                    }
                } else {
                    (None, None)
                }
            }
            // 2. 分离的 Occlusion 与 MetallicRoughness：启用 pack_orm 时离线合成为单张 ORM
            (Some(o), Some(m)) if self.pack_orm => {
                let occ_img = self.images.get(o);
                let mr_img = self.images.get(m);
                let rel_path = format!("textures/{}_orm_o{}_mr{}.png", self.stem, o, m);
                let abs_path = self.out_dir.join(&rel_path);
                match save_orm_png(occ_img, mr_img, &abs_path, self.max_texture_size) {
                    Ok(was_resized) => {
                        self.exported_count += 1;
                        self.orm_packed_count += 1;
                        if was_resized {
                            self.resized_count += 1;
                        }
                        (Some(rel_path.clone()), Some(rel_path))
                    }
                    Err(err) => {
                        eprintln!("  [警告] ORM 打包 (AO #{o} + MR #{m}) 失败，已跳过: {err}");
                        (None, None)
                    }
                }
            }
            // 3. 分离导出（--no-pack-orm）或仅存在单侧贴图
            (occ_opt, mr_opt) => {
                let occ_path = occ_opt.and_then(|o| {
                    let img = self.images.get(o)?;
                    let rel = format!("textures/{}_ao_{}.png", self.stem, o);
                    let abs = self.out_dir.join(&rel);
                    match save_occlusion_png(img, &abs, self.max_texture_size) {
                        Ok(was_resized) => {
                            self.exported_count += 1;
                            if was_resized {
                                self.resized_count += 1;
                            }
                            Some(rel)
                        }
                        Err(err) => {
                            eprintln!("  [警告] Occlusion 贴图 #{o} 导出失败，已跳过: {err}");
                            None
                        }
                    }
                });
                let mr_path = mr_opt.and_then(|m| {
                    let img = self.images.get(m)?;
                    let rel = format!("textures/{}_mr_{}.png", self.stem, m);
                    let abs = self.out_dir.join(&rel);
                    match save_orm_png(None, Some(img), &abs, self.max_texture_size) {
                        Ok(was_resized) => {
                            self.exported_count += 1;
                            if was_resized {
                                self.resized_count += 1;
                            }
                            Some(rel)
                        }
                        Err(err) => {
                            eprintln!("  [警告] MetallicRoughness 贴图 #{m} 导出失败，已跳过: {err}");
                            None
                        }
                    }
                });
                (occ_path, mr_path)
            }
        };

        self.orm_cache.insert(cache_key, result.clone());
        result
    }
}

#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
fn f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// 计算保持宽高比且不超过 `max_size` 的目标尺寸。
fn compute_target_dims(w: u32, h: u32, max_size: u32) -> (u32, u32, bool) {
    if max_size == 0 || (w <= max_size && h <= max_size) {
        return (w.max(1), h.max(1), false);
    }
    if w >= h {
        let new_h = ((h as u64 * max_size as u64 + (w as u64) / 2) / w.max(1) as u64).max(1) as u32;
        (max_size, new_h, true)
    } else {
        let new_w = ((w as u64 * max_size as u64 + (h as u64) / 2) / h.max(1) as u64).max(1) as u32;
        (new_w, max_size, true)
    }
}

/// 将 glTF 像素数据解包为标准化 `[0, 1]` 浮点通道（RGBA 四通道，缺失通道按默认填充），
/// 返回 `(pixels_rgba, source_channels, is_hdr_float)`。
fn decode_gltf_rgba_f32(
    data: &gltf::image::Data,
) -> Result<(Vec<[f32; 4]>, usize, bool), Box<dyn std::error::Error>> {
    let pixel_count = (data.width as usize) * (data.height as usize);
    let mut out = Vec::with_capacity(pixel_count);

    match data.format {
        Format::R8 => {
            if data.pixels.len() < pixel_count {
                return Err("R8 像素缓冲长度不足".into());
            }
            for &r in &data.pixels[..pixel_count] {
                let v = r as f32 / 255.0;
                out.push([v, v, v, 1.0]);
            }
            Ok((out, 1, false))
        }
        Format::R8G8 => {
            if data.pixels.len() < pixel_count * 2 {
                return Err("R8G8 像素缓冲长度不足".into());
            }
            for p in data.pixels[..pixel_count * 2].chunks_exact(2) {
                out.push([p[0] as f32 / 255.0, p[1] as f32 / 255.0, 0.0, 1.0]);
            }
            Ok((out, 2, false))
        }
        Format::R8G8B8 => {
            if data.pixels.len() < pixel_count * 3 {
                return Err("R8G8B8 像素缓冲长度不足".into());
            }
            for p in data.pixels[..pixel_count * 3].chunks_exact(3) {
                out.push([
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                    1.0,
                ]);
            }
            Ok((out, 3, false))
        }
        Format::R8G8B8A8 => {
            if data.pixels.len() < pixel_count * 4 {
                return Err("R8G8B8A8 像素缓冲长度不足".into());
            }
            for p in data.pixels[..pixel_count * 4].chunks_exact(4) {
                out.push([
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                    p[3] as f32 / 255.0,
                ]);
            }
            Ok((out, 4, false))
        }
        Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
            let ch = match data.format {
                Format::R16 => 1,
                Format::R16G16 => 2,
                Format::R16G16B16 => 3,
                Format::R16G16B16A16 => 4,
                _ => unreachable!(),
            };
            if data.pixels.len() < pixel_count * ch * 2 {
                return Err("R16* 像素缓冲长度不足".into());
            }
            let inv = 1.0 / 65535.0;
            for p in data.pixels[..pixel_count * ch * 2].chunks_exact(ch * 2) {
                let r = u16::from_le_bytes([p[0], p[1]]) as f32 * inv;
                let g = if ch >= 2 {
                    u16::from_le_bytes([p[2], p[3]]) as f32 * inv
                } else {
                    r
                };
                let b = if ch >= 3 {
                    u16::from_le_bytes([p[4], p[5]]) as f32 * inv
                } else if ch == 1 {
                    r
                } else {
                    0.0
                };
                let a = if ch >= 4 {
                    u16::from_le_bytes([p[6], p[7]]) as f32 * inv
                } else {
                    1.0
                };
                out.push([r, g, b, a]);
            }
            Ok((out, ch, false))
        }
        Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
            let ch = if matches!(data.format, Format::R32G32B32FLOAT) {
                3
            } else {
                4
            };
            if data.pixels.len() < pixel_count * ch * 4 {
                return Err("R32F 像素缓冲长度不足".into());
            }
            for p in data.pixels[..pixel_count * ch * 4].chunks_exact(ch * 4) {
                let r = f32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                let g = f32::from_le_bytes([p[4], p[5], p[6], p[7]]);
                let b = f32::from_le_bytes([p[8], p[9], p[10], p[11]]);
                let a = if ch == 4 {
                    f32::from_le_bytes([p[12], p[13], p[14], p[15]])
                } else {
                    1.0
                };
                out.push([r, g, b, a]);
            }
            Ok((out, ch, true))
        }
    }
}

/// 可分离 2D 重采样（降采样采用精确像素覆盖面积加权平均，升采样采用双线性插值），
/// 在给定浮点向量空间（Linear RGB / 切线空间 Normal / 线性 ORM）中执行，避免高频走样。
fn resample_f32_grid<const C: usize>(
    src: &[[f32; C]],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<[f32; C]> {
    let (sw, sh, dw, dh) = (src_w as usize, src_h as usize, dst_w as usize, dst_h as usize);
    if sw == dw && sh == dh {
        return src.to_vec();
    }

    // 1. 水平重采样: sw x sh -> dw x sh
    let mut temp = vec![[0.0f32; C]; dw * sh];
    if dw == sw {
        temp.copy_from_slice(src);
    } else if dw < sw {
        let scale = sw as f32 / dw as f32;
        for dx in 0..dw {
            let x0 = dx as f32 * scale;
            let x1 = (dx + 1) as f32 * scale;
            let ix0 = (x0.floor() as usize).min(sw - 1);
            let ix1 = ((x1.ceil() as usize).saturating_sub(1)).min(sw - 1);
            for y in 0..sh {
                let mut acc = [0.0f32; C];
                let mut w_sum = 0.0f32;
                let row = y * sw;
                for sx in ix0..=ix1 {
                    let left = (sx as f32).max(x0);
                    let right = ((sx + 1) as f32).min(x1);
                    let w = (right - left).max(0.0);
                    if w > 0.0 {
                        let p = src[row + sx];
                        for c in 0..C {
                            acc[c] += p[c] * w;
                        }
                        w_sum += w;
                    }
                }
                let inv = if w_sum > 0.0 { 1.0 / w_sum } else { 0.0 };
                for c in 0..C {
                    acc[c] *= inv;
                }
                temp[y * dw + dx] = acc;
            }
        }
    } else {
        let scale = sw as f32 / dw as f32;
        for dx in 0..dw {
            let fx = (dx as f32 + 0.5) * scale - 0.5;
            let x0 = fx.floor().clamp(0.0, (sw - 1) as f32) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let t = (fx - x0 as f32).clamp(0.0, 1.0);
            for y in 0..sh {
                let p0 = src[y * sw + x0];
                let p1 = src[y * sw + x1];
                let mut out_p = [0.0f32; C];
                for c in 0..C {
                    out_p[c] = p0[c] * (1.0 - t) + p1[c] * t;
                }
                temp[y * dw + dx] = out_p;
            }
        }
    }

    // 2. 垂直重采样: dw x sh -> dw x dh
    if dh == sh {
        return temp;
    }
    let mut dst = vec![[0.0f32; C]; dw * dh];
    if dh < sh {
        let scale = sh as f32 / dh as f32;
        for dy in 0..dh {
            let y0 = dy as f32 * scale;
            let y1 = (dy + 1) as f32 * scale;
            let iy0 = (y0.floor() as usize).min(sh - 1);
            let iy1 = ((y1.ceil() as usize).saturating_sub(1)).min(sh - 1);
            for x in 0..dw {
                let mut acc = [0.0f32; C];
                let mut w_sum = 0.0f32;
                for sy in iy0..=iy1 {
                    let top = (sy as f32).max(y0);
                    let bot = ((sy + 1) as f32).min(y1);
                    let w = (bot - top).max(0.0);
                    if w > 0.0 {
                        let p = temp[sy * dw + x];
                        for c in 0..C {
                            acc[c] += p[c] * w;
                        }
                        w_sum += w;
                    }
                }
                let inv = if w_sum > 0.0 { 1.0 / w_sum } else { 0.0 };
                for c in 0..C {
                    acc[c] *= inv;
                }
                dst[dy * dw + x] = acc;
            }
        }
    } else {
        let scale = sh as f32 / dh as f32;
        for dy in 0..dh {
            let fy = (dy as f32 + 0.5) * scale - 0.5;
            let y0 = fy.floor().clamp(0.0, (sh - 1) as f32) as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let t = (fy - y0 as f32).clamp(0.0, 1.0);
            for x in 0..dw {
                let p0 = temp[y0 * dw + x];
                let p1 = temp[y1 * dw + x];
                let mut out_p = [0.0f32; C];
                for c in 0..C {
                    out_p[c] = p0[c] * (1.0 - t) + p1[c] * t;
                }
                dst[dy * dw + x] = out_p;
            }
        }
    }

    dst
}

/// 导出 sRGB 颜色贴图（BaseColor / Emissive）：在线性空间完成高质量重采样后回写 sRGB。
fn save_srgb_color_png(
    data: &gltf::image::Data,
    out_path: &Path,
    max_texture_size: u32,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (dst_w, dst_h, resized) = compute_target_dims(data.width, data.height, max_texture_size);
    let (mut pixels, src_channels, is_hdr_float) = decode_gltf_rgba_f32(data)?;

    // 仅在需要重采样时才执行 sRGB <-> Linear 往返，保持未缩放原图的逐字节保真
    if resized {
        if !is_hdr_float {
            for p in &mut pixels {
                p[0] = srgb_to_linear(p[0]);
                p[1] = srgb_to_linear(p[1]);
                p[2] = srgb_to_linear(p[2]);
            }
        }
        pixels = resample_f32_grid(&pixels, data.width, data.height, dst_w, dst_h);
        for p in &mut pixels {
            p[0] = linear_to_srgb(p[0]);
            p[1] = linear_to_srgb(p[1]);
            p[2] = linear_to_srgb(p[2]);
        }
    } else if is_hdr_float {
        for p in &mut pixels {
            p[0] = linear_to_srgb(p[0]);
            p[1] = linear_to_srgb(p[1]);
            p[2] = linear_to_srgb(p[2]);
        }
    }

    let has_non_opaque_alpha = src_channels == 4 && pixels.iter().any(|p| f32_to_u8(p[3]) < 255);
    if has_non_opaque_alpha {
        let mut raw = Vec::with_capacity((dst_w * dst_h * 4) as usize);
        for p in &pixels {
            raw.extend_from_slice(&[
                f32_to_u8(p[0]),
                f32_to_u8(p[1]),
                f32_to_u8(p[2]),
                f32_to_u8(p[3]),
            ]);
        }
        let buf = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(dst_w, dst_h, raw)
            .ok_or("sRGB RGBA8 缓冲构造失败")?;
        buf.save_with_format(out_path, image::ImageFormat::Png)?;
    } else {
        let mut raw = Vec::with_capacity((dst_w * dst_h * 3) as usize);
        for p in &pixels {
            raw.extend_from_slice(&[f32_to_u8(p[0]), f32_to_u8(p[1]), f32_to_u8(p[2])]);
        }
        let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(dst_w, dst_h, raw)
            .ok_or("sRGB RGB8 缓冲构造失败")?;
        buf.save_with_format(out_path, image::ImageFormat::Png)?;
    }

    Ok(resized)
}

/// 导出切线空间法线贴图：
/// - 2 通道 (`R8G8`/`R16G16`) 自动推导 `z = sqrt(1 - x^2 - y^2)`；
/// - 缩放后逐像素重归一化，保证法线方向与高光响应不失真。
fn save_normal_png(
    data: &gltf::image::Data,
    out_path: &Path,
    max_texture_size: u32,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (dst_w, dst_h, resized) = compute_target_dims(data.width, data.height, max_texture_size);
    let (rgba, src_channels, _) = decode_gltf_rgba_f32(data)?;

    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(rgba.len());
    for p in &rgba {
        let n = if src_channels == 2 {
            let nx = p[0] * 2.0 - 1.0;
            let ny = p[1] * 2.0 - 1.0;
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            [nx, ny, nz]
        } else if src_channels == 1 {
            [0.0, 0.0, 1.0]
        } else {
            [p[0] * 2.0 - 1.0, p[1] * 2.0 - 1.0, p[2] * 2.0 - 1.0]
        };
        normals.push(n);
    }

    if resized {
        normals = resample_f32_grid(&normals, data.width, data.height, dst_w, dst_h);
    }

    let mut raw = Vec::with_capacity((dst_w * dst_h * 3) as usize);
    for n in &normals {
        let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        let unit = if len_sq > 1e-8 {
            let inv = len_sq.sqrt().recip();
            [n[0] * inv, n[1] * inv, n[2] * inv]
        } else {
            [0.0, 0.0, 1.0]
        };
        raw.extend_from_slice(&[
            f32_to_u8(unit[0] * 0.5 + 0.5),
            f32_to_u8(unit[1] * 0.5 + 0.5),
            f32_to_u8(unit[2] * 0.5 + 0.5),
        ]);
    }

    let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(dst_w, dst_h, raw)
        .ok_or("Normal RGB8 缓冲构造失败")?;
    buf.save_with_format(out_path, image::ImageFormat::Png)?;
    Ok(resized)
}

/// 导出独立的单通道 Occlusion 贴图（线性空间滤波，Luma8 PNG）。
fn save_occlusion_png(
    data: &gltf::image::Data,
    out_path: &Path,
    max_texture_size: u32,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (dst_w, dst_h, resized) = compute_target_dims(data.width, data.height, max_texture_size);
    let (rgba, _, _) = decode_gltf_rgba_f32(data)?;
    let mut occ: Vec<[f32; 1]> = rgba.iter().map(|p| [p[0]]).collect();
    if resized {
        occ = resample_f32_grid(&occ, data.width, data.height, dst_w, dst_h);
    }
    let raw: Vec<u8> = occ.iter().map(|p| f32_to_u8(p[0])).collect();
    let buf = image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(dst_w, dst_h, raw)
        .ok_or("Occlusion Luma8 缓冲构造失败")?;
    buf.save_with_format(out_path, image::ImageFormat::Png)?;
    Ok(resized)
}

/// 导出标准 glTF 3 通道 ORM 贴图（R=Occlusion, G=Roughness, B=Metallic），线性空间滤波。
/// 支持：
/// - `occ` 与 `mr` 指向同一张图（原生 ORM 透传与线性缩放）；
/// - `occ` 与 `mr` 指向两张不同尺寸的图（自动对齐到两者最大分辨率并受 `max_texture_size` 限制后打包）；
/// - 仅传入 `mr`（`occ = None` 时 R 通道填 1.0；若 `mr` 为 2 通道 `R8G8`/`R16G16`，自动将 R/G 映射至标准 G=Roughness, B=Metallic）。
fn save_orm_png(
    occ_data: Option<&gltf::image::Data>,
    mr_data: Option<&gltf::image::Data>,
    out_path: &Path,
    max_texture_size: u32,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (base_w, base_h) = match (occ_data, mr_data) {
        (Some(o), Some(m)) => (o.width.max(m.width), o.height.max(m.height)),
        (Some(o), None) => (o.width, o.height),
        (None, Some(m)) => (m.width, m.height),
        (None, None) => return Err("ORM 导出缺少输入图像".into()),
    };
    let (dst_w, dst_h, resized) = compute_target_dims(base_w, base_h, max_texture_size);
    let dst_len = (dst_w as usize) * (dst_h as usize);

    // 同源 ORM 快捷路径
    if let (Some(o), Some(m)) = (occ_data, mr_data)
        && std::ptr::eq(o, m)
    {
        let (rgba, ch, _) = decode_gltf_rgba_f32(o)?;
        let mut orm: Vec<[f32; 3]> = rgba
            .iter()
            .map(|p| {
                if ch == 2 {
                    [1.0, p[0], p[1]]
                } else if ch == 1 {
                    [p[0], p[0], 0.0]
                } else {
                    [p[0], p[1], p[2]]
                }
            })
            .collect();
        if dst_w != o.width || dst_h != o.height {
            orm = resample_f32_grid(&orm, o.width, o.height, dst_w, dst_h);
        }
        let mut raw = Vec::with_capacity(dst_len * 3);
        for p in &orm {
            raw.extend_from_slice(&[f32_to_u8(p[0]), f32_to_u8(p[1]), f32_to_u8(p[2])]);
        }
        let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(dst_w, dst_h, raw)
            .ok_or("ORM RGB8 缓冲构造失败")?;
        buf.save_with_format(out_path, image::ImageFormat::Png)?;
        return Ok(resized);
    }

    // 分别提取 O（R 通道）与 MR（G=Roughness, B=Metallic）并重采样到统一目标尺寸
    let occ_channel: Vec<[f32; 1]> = if let Some(o) = occ_data {
        let (rgba, _, _) = decode_gltf_rgba_f32(o)?;
        let src: Vec<[f32; 1]> = rgba.iter().map(|p| [p[0]]).collect();
        resample_f32_grid(&src, o.width, o.height, dst_w, dst_h)
    } else {
        vec![[1.0]; dst_len]
    };

    let mr_channels: Vec<[f32; 2]> = if let Some(m) = mr_data {
        let (rgba, ch, _) = decode_gltf_rgba_f32(m)?;
        let src: Vec<[f32; 2]> = rgba
            .iter()
            .map(|p| {
                if ch == 2 {
                    // 2 通道 MR（R=Roughness, G=Metallic）规范化到标准 G/B 槽位
                    [p[0], p[1]]
                } else if ch == 1 {
                    [p[0], 0.0]
                } else {
                    // glTF 标准：G=Roughness, B=Metallic
                    [p[1], p[2]]
                }
            })
            .collect();
        resample_f32_grid(&src, m.width, m.height, dst_w, dst_h)
    } else {
        vec![[1.0, 1.0]; dst_len]
    };

    let mut raw = Vec::with_capacity(dst_len * 3);
    for i in 0..dst_len {
        raw.extend_from_slice(&[
            f32_to_u8(occ_channel[i][0]),
            f32_to_u8(mr_channels[i][0]),
            f32_to_u8(mr_channels[i][1]),
        ]);
    }
    let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(dst_w, dst_h, raw)
        .ok_or("打包 ORM RGB8 缓冲构造失败")?;
    buf.save_with_format(out_path, image::ImageFormat::Png)?;
    Ok(resized)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_test_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "makemeshlet_tex_test_{}_{}",
            tag,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_srgb_linear_downsample() {
        // 2x2 黑白棋盘格 (0 与 255)，在线性空间均值 0.5 -> sRGB 约 188（而非错误 gamma 均值 128）
        let img = gltf::image::Data {
            pixels: vec![
                0, 0, 0, 255, 255, 255, // row 0
                255, 255, 255, 0, 0, 0, // row 1
            ],
            format: Format::R8G8B8,
            width: 2,
            height: 2,
        };
        let dir = temp_test_dir("srgb");
        let out = dir.join("srgb.png");
        let resized = save_srgb_color_png(&img, &out, 1).unwrap();
        assert!(resized);

        let loaded = image::open(&out).unwrap().to_rgb8();
        assert_eq!(loaded.dimensions(), (1, 1));
        let px = loaded.get_pixel(0, 0).0;
        assert!((px[0] as i32 - 188).abs() <= 2, "got {:?}", px);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_two_channel_normal_reconstruct_and_renormalize() {
        // 2x1 2-channel 法线：(nx=+0.6, ny=0 -> nz=+0.8) 与 (nx=-0.6, ny=0 -> nz=+0.8)
        // 降采样到 1x1 后平均向量为 (0, 0, 0.8)，重归一化应得 (0, 0, 1) -> RGB (128, 128, 255)
        let c_pos = f32_to_u8(0.6 * 0.5 + 0.5);
        let c_neg = f32_to_u8(-0.6 * 0.5 + 0.5);
        let c_zero = f32_to_u8(0.5);
        let img = gltf::image::Data {
            pixels: vec![c_pos, c_zero, c_neg, c_zero],
            format: Format::R8G8,
            width: 2,
            height: 1,
        };
        let dir = temp_test_dir("normal");
        let out = dir.join("normal.png");
        let resized = save_normal_png(&img, &out, 1).unwrap();
        assert!(resized);

        let loaded = image::open(&out).unwrap().to_rgb8();
        assert_eq!(loaded.dimensions(), (1, 1));
        let px = loaded.get_pixel(0, 0).0;
        assert!((px[0] as i32 - 128).abs() <= 1, "got {:?}", px);
        assert!((px[1] as i32 - 128).abs() <= 1, "got {:?}", px);
        assert_eq!(px[2], 255, "Z 应重归一化至 1.0，实际 {:?}", px);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pack_separate_ao_and_rg_mr_into_orm() {
        // AO (R8, 1x1) = 100, MR (R8G8, 2x2) = (Roughness=200, Metallic=50)
        // 合并为 2x2 ORM 后每像素应为 R=100, G=200, B=50，且两个槽位返回同一相对路径
        let ao_img = gltf::image::Data {
            pixels: vec![100],
            format: Format::R8,
            width: 1,
            height: 1,
        };
        let mr_img = gltf::image::Data {
            pixels: vec![200, 50, 200, 50, 200, 50, 200, 50],
            format: Format::R8G8,
            width: 2,
            height: 2,
        };
        let images = vec![ao_img, mr_img];
        let dir = temp_test_dir("orm");
        let mut exporter = SemanticTextureExporter::new(&images, &dir, "scene", 1024, true, true);
        let (occ_path, mr_path) = exporter.export_orm_pair(Some(0), Some(1));
        assert!(occ_path.is_some());
        assert_eq!(occ_path, mr_path);
        assert_eq!(exporter.orm_packed_count, 1);

        let loaded = image::open(dir.join(occ_path.unwrap())).unwrap().to_rgb8();
        assert_eq!(loaded.dimensions(), (2, 2));
        for p in loaded.pixels() {
            assert_eq!(p.0, [100, 200, 50]);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
