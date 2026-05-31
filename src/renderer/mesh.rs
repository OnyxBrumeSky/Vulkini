// src/renderer/mesh.rs
use std::sync::Arc;
use vulkano::buffer::CpuBufferPool;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use nalgebra_glm::{rotate_normalized_axis, vec3, TMat4};

use crate::camera::LightController;
use crate::scene_object::{SceneObject, Vertex};
use crate::shader::{fs, vs};
use crate::texture::GpuTexture;
use vulkano::buffer::{CpuAccessibleBuffer, TypedBufferAccess};

pub struct MeshRenderer {
    pipeline: Arc<GraphicsPipeline>,
    uniform_buffer: CpuBufferPool<vs::ty::MVP_Data>,
    ambient_buffer: CpuBufferPool<fs::ty::Ambient_Data>,
    directional_buffer: CpuBufferPool<fs::ty::Directional_Light_Data>,
}

impl MeshRenderer {
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        subpass: Subpass,
    ) -> Self {
        let vs_shader = vs::load(device.clone()).unwrap();
        let fs_shader = fs::load(device.clone()).unwrap();

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .render_pass(subpass)
            .build(device)
            .unwrap();

        let uniform_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let ambient_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let directional_buffer = CpuBufferPool::uniform_buffer(memory_allocator);

        Self {
            pipeline,
            uniform_buffer,
            ambient_buffer,
            directional_buffer,
        }
    }

    pub fn draw(
        &self,
        cmd: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        ds_allocator: &StandardDescriptorSetAllocator,
        objects: &[SceneObject],
        vertex_buffers: &[Arc<CpuAccessibleBuffer<[Vertex]>>],
        projection: &TMat4<f32>,
        view_matrix: &TMat4<f32>,
        ambient_color: [f32; 3],
        ambient_intensity: f32,
        light: &LightController,
        white_texture: &Arc<GpuTexture>,
        elapsed_time: f32,
    ) {
        cmd.bind_pipeline_graphics(self.pipeline.clone());

        // Génération des subbuffers communs à tous les objets pour cette frame
        let ambient_subbuffer = self.ambient_buffer.from_data(fs::ty::Ambient_Data {
            color: ambient_color.into(),
            intensity: ambient_intensity.into(),
        }).unwrap();

        let directional_subbuffer = self.directional_buffer.from_data(fs::ty::Directional_Light_Data {
            position: light.position_vec4().into(),
            color: light.effective_color().into(),
        }).unwrap();

        let mesh_layout = self.pipeline.layout().set_layouts().get(0).unwrap().clone();

        for (obj, vbuf) in objects.iter().zip(vertex_buffers.iter()) {
            // Calcul de la matrice de modèle de l'objet
            let angle = elapsed_time * obj.rotation_speed;
            let model = rotate_normalized_axis(&obj.base_transform, angle, &vec3(0.0, 1.0, 0.0));

            let uniform_subbuffer = self.uniform_buffer.from_data(vs::ty::MVP_Data {
                model: model.into(),
                view: (*view_matrix).into(),
                projection: (*projection).into(),
            }).unwrap();

            // Choix de la texture
            let tex = obj.texture.as_ref().unwrap_or(white_texture);

            let mesh_set = PersistentDescriptorSet::new(
                ds_allocator,
                mesh_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform_subbuffer),
                    WriteDescriptorSet::buffer(1, ambient_subbuffer.clone()),
                    WriteDescriptorSet::buffer(2, directional_subbuffer.clone()),
                    WriteDescriptorSet::image_view_sampler(3, tex.view.clone(), tex.sampler.clone()),
                ],
            )
            .unwrap();

            cmd.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                mesh_set,
            )
            .bind_vertex_buffers(0, vbuf.clone())
            .draw(vbuf.len() as u32, 1, 0, 0)
            .unwrap();
        }
    }
}