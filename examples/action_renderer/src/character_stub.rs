use bevy::prelude::*;

/// 用于验证“Meshlet 静态高模 + 标准 Mesh3d 动态角色/透明 VFX”混合管线的测试桩模块。
/// 后续可直接替换为真实的骨骼动画角色、武器刀光与受击材质。
pub struct CharacterActionStubPlugin;

#[derive(Component)]
struct TestActionActor {
    base_pos: Vec3,
}

impl Plugin for CharacterActionStubPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_character_and_vfx_stub)
            .add_systems(Update, animate_actor_stub);
    }
}

fn spawn_character_and_vfx_stub(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // 地面参考平面（标准 Mesh3d，接收来自 Meshlet 与角色的阴影以及 IrradianceVolume 间接光）
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(20.0, 20.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.22, 0.23, 0.26),
            perceptual_roughness: 0.85,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.22, 0.0),
    ));

    // 动态角色代理（标准 Opaque Mesh3d，测试与 Meshlet 场景的深度遮挡及 IrradianceVolume 一致性）
    let actor_pos = Vec3::new(2.2, 0.45, 0.5);
    commands
        .spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.28, 0.75))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.85, 0.25, 0.2),
                perceptual_roughness: 0.4,
                metallic: 0.1,
                ..default()
            })),
            Transform::from_translation(actor_pos),
            TestActionActor {
                base_pos: actor_pos,
            },
        ))
        .with_children(|parent| {
            // 透明刀光/能量核心代理（测试 Meshlet 不支持的 AlphaBlend 透明物体在混合管线下的渲染）
            parent.spawn((
                Mesh3d(meshes.add(Torus::new(0.35, 0.04))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(0.2, 0.8, 1.0, 0.45),
                    emissive: LinearRgba::rgb(0.4, 2.0, 3.5),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    ..default()
                })),
                Transform::from_xyz(0.0, 0.2, 0.0)
                    .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_4)),
            ));
        });
}

fn animate_actor_stub(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &TestActionActor)>,
) {
    let t = time.elapsed_secs();
    for (mut tf, actor) in &mut query {
        tf.translation = actor.base_pos + Vec3::new(0.0, (t * 2.0).sin() * 0.15, (t * 1.2).cos() * 0.6);
        tf.rotate_y(time.delta_secs() * 1.5);
    }
}
