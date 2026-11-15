use bevy::{
    asset::{io::Reader, AssetLoader, LoadContext},
    light::IrradianceVolume,
    prelude::*,
};
use serde::Deserialize;
use thiserror::Error;

/// `mutbaker-gui` 输出的 `.meta.json` 元数据。
#[derive(Asset, TypePath, Debug, Clone, Deserialize)]
pub struct IrradianceVolumeMeta {
    #[serde(default)]
    pub bounds_min: Option<[f32; 3]>,
    #[serde(default)]
    pub bounds_max: Option<[f32; 3]>,
    #[serde(default)]
    pub bevy_transform: Option<IrradianceVolumeMetaTransform>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IrradianceVolumeMetaTransform {
    pub translation: [f32; 3],
    pub scale: [f32; 3],
}

impl IrradianceVolumeMeta {
    pub fn transform(&self) -> Option<Transform> {
        if let (Some(bounds_min), Some(bounds_max)) = (self.bounds_min, self.bounds_max) {
            let min = Vec3::from_array(bounds_min);
            let max = Vec3::from_array(bounds_max);
            return Some(
                Transform::from_translation((min + max) * 0.5).with_scale(max - min),
            );
        }

        self.bevy_transform.as_ref().map(|t| {
            Transform::from_translation(Vec3::from_array(t.translation))
                .with_scale(Vec3::from_array(t.scale))
        })
    }
}

#[derive(Default, TypePath)]
pub struct IrradianceVolumeMetaLoader;

#[derive(Debug, Error)]
pub enum IrradianceVolumeMetaLoaderError {
    #[error("Failed to read irradiance volume metadata: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse irradiance volume metadata: {0}")]
    Json(#[from] serde_json::Error),
}

impl AssetLoader for IrradianceVolumeMetaLoader {
    type Asset = IrradianceVolumeMeta;
    type Settings = ();
    type Error = IrradianceVolumeMetaLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn extensions(&self) -> &[&str] {
        &["meta.json"]
    }
}

#[derive(Resource, Clone)]
pub struct IrradianceConfig {
    pub ktx2_asset_path: String,
    pub meta_asset_path: String,
    pub intensity: f32,
}

#[derive(Resource)]
struct IrradianceHandles {
    voxels: Handle<Image>,
    meta: Handle<IrradianceVolumeMeta>,
    spawned: bool,
    enabled: bool,
    cached_transform: Option<Transform>,
}

pub struct IrradianceScenePlugin;

impl Plugin for IrradianceScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<IrradianceVolumeMeta>()
            .init_asset_loader::<IrradianceVolumeMetaLoader>()
            .add_systems(Startup, load_irradiance_assets)
            .add_systems(Update, (spawn_irradiance_when_ready, toggle_irradiance_volume));
    }
}

fn load_irradiance_assets(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    config: Res<IrradianceConfig>,
) {
    let voxels = asset_server.load(&config.ktx2_asset_path);
    let meta = asset_server.load(&config.meta_asset_path);
    commands.insert_resource(IrradianceHandles {
        voxels,
        meta,
        spawned: false,
        enabled: true,
        cached_transform: None,
    });
}

fn spawn_irradiance_when_ready(
    mut commands: Commands,
    mut handles: ResMut<IrradianceHandles>,
    metas: Res<Assets<IrradianceVolumeMeta>>,
    config: Res<IrradianceConfig>,
) {
    if handles.spawned {
        return;
    }
    let Some(meta) = metas.get(&handles.meta) else {
        return;
    };
    let Some(transform) = meta.transform() else {
        warn!("IrradianceVolume meta 缺少 bounds_min/bounds_max 或 bevy_transform，跳过生成");
        handles.spawned = true;
        return;
    };

    info!(
        "生成 IrradianceVolume: translation={:?}, scale={:?}, intensity={}",
        transform.translation, transform.scale, config.intensity
    );

    commands.spawn((
        transform,
        IrradianceVolume {
            voxels: handles.voxels.clone(),
            intensity: config.intensity,
            ..default()
        },
    ));
    handles.cached_transform = Some(transform);
    handles.spawned = true;
}

fn toggle_irradiance_volume(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut handles: ResMut<IrradianceHandles>,
    volumes: Query<Entity, With<IrradianceVolume>>,
    config: Res<IrradianceConfig>,
) {
    if !keys.just_pressed(KeyCode::Space) {
        return;
    }
    let Some(transform) = handles.cached_transform else {
        return;
    };

    handles.enabled = !handles.enabled;
    if handles.enabled {
        if volumes.is_empty() {
            commands.spawn((
                transform,
                IrradianceVolume {
                    voxels: handles.voxels.clone(),
                    intensity: config.intensity,
                    ..default()
                },
            ));
            info!("IrradianceVolume: ON");
        }
    } else {
        for entity in &volumes {
            commands.entity(entity).despawn();
        }
        info!("IrradianceVolume: OFF");
    }
}
