// src/scene_object.rs
use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use nalgebra_glm::{identity, TMat4};
use crate::texture::GpuTexture;

// ─── Vertex ──────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub tex_coords: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position, normal, color, tex_coords);

// ─── Objet de scène ───────────────────────────────────────────────────────────

pub struct SceneObject {
    pub vertices: Vec<Vertex>,
    pub base_transform: TMat4<f32>,
    pub rotation_speed: f32,
    pub texture: Option<Arc<GpuTexture>>,
}

impl SceneObject {
    pub fn new(vertices: Vec<Vertex>, base_transform: TMat4<f32>) -> Self {
        Self {
            vertices,
            base_transform,
            rotation_speed: 0.0,
            texture: None,
        }
    }

    pub fn with_rotation(mut self, speed: f32) -> Self {
        self.rotation_speed = speed;
        self
    }

    pub fn with_texture(mut self, tex: Arc<GpuTexture>) -> Self {
        self.texture = Some(tex);
        self
    }
}

// ─── Uniformes partagés (Requis par renderer/mod.rs) ──────────────────────────

#[derive(Default, Debug, Clone, Copy)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct MVP {
    pub model: TMat4<f32>,
    pub view: TMat4<f32>,
    pub projection: TMat4<f32>,
}

impl MVP {
    pub fn new() -> Self {
        MVP {
            model: identity(),
            view: identity(),
            projection: identity(),
        }
    }
}