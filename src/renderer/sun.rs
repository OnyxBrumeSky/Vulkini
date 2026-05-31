// src/renderer/sun.rs
use std::sync::Arc;
use vulkano::buffer::CpuBufferPool;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use nalgebra_glm::TMat4;

use crate::camera::LightController;
use crate::shader::{sun_fs, sun_vs};

pub struct SunRenderer {
    pipeline: Arc<GraphicsPipeline>,
    sun_data_buffer: CpuBufferPool<sun_vs::ty::Sun_Data>,
    sun_color_buffer: CpuBufferPool<sun_fs::ty::Sun_Color>,
}

impl SunRenderer {
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        subpass: Subpass,
    ) -> Self {
        let sun_vs_shader = sun_vs::load(device.clone()).unwrap();
        let sun_fs_shader = sun_fs::load(device.clone()).unwrap();

        let sun_blend = ColorBlendState::new(1).blend(AttachmentBlend {
            color_op: BlendOp::Add,
            color_source: BlendFactor::SrcAlpha,
            color_destination: BlendFactor::OneMinusSrcAlpha,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::One,
            alpha_destination: BlendFactor::OneMinusSrcAlpha,
        });

        use vulkano::pipeline::StateMode;
        let sun_depth = DepthStencilState {
            depth: Some(DepthState {
                enable_dynamic: false,
                compare_op: StateMode::Fixed(CompareOp::Less),
                write_enable: StateMode::Fixed(false),
            }),
            depth_bounds: None,
            stencil: None,
        };

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new())
            .vertex_shader(sun_vs_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip))
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(sun_fs_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(sun_depth)
            .color_blend_state(sun_blend)
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .render_pass(subpass)
            .build(device)
            .unwrap();

        let sun_data_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let sun_color_buffer = CpuBufferPool::uniform_buffer(memory_allocator);

        Self {
            pipeline,
            sun_data_buffer,
            sun_color_buffer,
        }
    }

    pub fn draw(
        &self,
        cmd: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        ds_allocator: &StandardDescriptorSetAllocator,
        projection: &TMat4<f32>,
        view_matrix: &TMat4<f32>,
        light: &LightController,
    ) {
        let sun_data_sub = {
            let p = light.position;
            let data = sun_vs::ty::Sun_Data {
                world_pos: [p.x, p.y, p.z],
                size: 0.6,
                view: (*view_matrix).into(),
                projection: (*projection).into(),
            };
            self.sun_data_buffer.from_data(data).unwrap()
        };

        let sun_color_sub = {
            let ec = light.effective_color();
            let data = sun_fs::ty::Sun_Color {
                core_color: [1.0, 0.97, 0.88],
                _pad0: 0.0,
                halo_color: ec,
                _pad1: 0.0,
            };
            self.sun_color_buffer.from_data(data).unwrap()
        };

        let sun_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let sun_set = PersistentDescriptorSet::new(
            ds_allocator,
            sun_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, sun_data_sub),
                WriteDescriptorSet::buffer(1, sun_color_sub),
            ],
        )
        .unwrap();

        cmd.bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                sun_set,
            )
            .draw(4, 1, 0, 0)
            .unwrap();
    }
}