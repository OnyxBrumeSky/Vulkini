// src/renderer/skybox.rs
use std::sync::Arc;
use vulkano::buffer::CpuBufferPool;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use nalgebra_glm::TMat4;

use crate::shader::sky_fs2;
use crate::shader::sky_vs2;

pub struct SkySettings {
    pub sky_color: [f32; 3],
    pub ground_color: [f32; 3],
    pub sharpness: f32,
}

impl Default for SkySettings {
    fn default() -> Self {
        Self {
            sky_color: [0.1, 0.3, 0.7],
            ground_color: [0.1, 0.1, 0.1],
            sharpness: 2.0,
        }
    }
}

pub struct SkyboxRenderer {
    pipeline: Arc<GraphicsPipeline>,
    sky_data_buffer: CpuBufferPool<sky_fs2::ty::Sky_Data>,
    pub settings: SkySettings,
}

impl SkyboxRenderer {
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        subpass: Subpass,
    ) -> Self {
        let sky_vs_shader = sky_vs2::load(device.clone()).unwrap();
        let sky_fs_shader = sky_fs2::load(device.clone()).unwrap();

        // Configuration du test de profondeur spécifique pour la Skybox
        // On utilise LessOrEqual et l'écriture est désactivée afin que la skybox 
        // se dessine derrière tous les futurs objets.
        use vulkano::pipeline::StateMode;
        let sky_depth = DepthStencilState {
            depth: Some(DepthState {
                enable_dynamic: false,
                compare_op: StateMode::Fixed(CompareOp::LessOrEqual),
                write_enable: StateMode::Fixed(false),
            }),
            depth_bounds: None,
            stencil: None,
        };

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new()) // Pas de vertex buffers, générés dans le shader
            .vertex_shader(sky_vs_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(sky_fs_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(sky_depth)
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .render_pass(subpass)
            .build(device)
            .unwrap();

        let sky_data_buffer = CpuBufferPool::uniform_buffer(memory_allocator);

        Self {
            pipeline,
            sky_data_buffer,
            settings: SkySettings::default(),
        }
    }

    pub fn draw(
        &self,
        cmd: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        ds_allocator: &StandardDescriptorSetAllocator,
        projection: &TMat4<f32>,
        view_matrix: &TMat4<f32>,
    ) {
        // Calcul de la matrice de projection-vue inverse requise par ton shader de ciel
        let proj_view = projection * view_matrix;
        let inv_proj_view = nalgebra_glm::inverse(&proj_view);

        let sky_data_sub = {
            let data = sky_fs2::ty::Sky_Data {
                sky_color: self.settings.sky_color.into(),
                sharpness: self.settings.sharpness,
                ground_color: self.settings.ground_color.into(),
                _pad: 0.0,
                inv_proj_view: inv_proj_view.into(),
            };
            self.sky_data_buffer.from_data(data).unwrap()
        };

        let sky_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let sky_set = PersistentDescriptorSet::new(
            ds_allocator,
            sky_layout.clone(),
            [WriteDescriptorSet::buffer(0, sky_data_sub)],
        )
        .unwrap();

        cmd.bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                sky_set,
            )
            // On dessine un unique triangle à 3 sommets qui couvre tout l'écran
            .draw(3, 1, 0, 0)
            .unwrap();
    }
}