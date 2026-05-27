// vulkini.rs
use bytemuck::{Pod, Zeroable};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::{Version, VulkanLibrary};

use vulkano_win::VkSurfaceBuild;

use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use nalgebra_glm::{
    half_pi, identity, perspective, pi, rotate_normalized_axis, translate, vec3, TMat4,
};

use std::sync::Arc;
use std::time::Instant;

use crate::camera::{Camera, LightController};
use crate::shader::{fs, sky_fs2, sky_vs2, sun_fs, sun_vs, vs};

// ─── Vertex ──────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, normal, color);

// ─── Uniformes internes ───────────────────────────────────────────────────────

#[derive(Default, Debug, Clone)]
struct AmbientLight {
    color: [f32; 3],
    intensity: f32,
}

#[derive(Debug, Clone)]
struct MVP {
    model: TMat4<f32>,
    view: TMat4<f32>,
    projection: TMat4<f32>,
}

impl MVP {
    fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            projection: identity(),
        }
    }
}

// ─── Données skybox (envoyées en uniform) ────────────────────────────────────

/// Paramètres de la skybox, modifiables à la volée si besoin.
pub struct SkySettings {
    /// Couleur du ciel
    pub sky_color: [f32; 3],
    /// Couleur du sol
    pub ground_color: [f32; 3],
    /// Netteté de la transition ciel/sol. Plus grand = ligne plus nette.
    /// 8.0 donne une transition douce, 30.0 une ligne franche.
    pub sharpness: f32,
}

impl Default for SkySettings {
    fn default() -> Self {
        Self {
            sky_color: [0.18, 0.45, 0.82],    // bleu ciel
            ground_color: [0.08, 0.06, 0.05], // sol sombre
            sharpness: 8.0,
        }
    }
}

// ─── Structure principale ────────────────────────────────────────────────────

pub struct Vulkmini {
    event_loop: EventLoop<()>,
    surface: Arc<Surface>,
    mvp: MVP,
    swapchain: Arc<Swapchain>,
    framebuffers: Vec<Arc<Framebuffer>>,
    memory_allocator: Arc<
        vulkano::memory::allocator::GenericMemoryAllocator<
            Arc<vulkano::memory::allocator::FreeListAllocator>,
        >,
    >,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    camera: Camera,
    // Mesh
    uniform_buffer: CpuBufferPool<vs::ty::MVP_Data>,
    ambient_light: AmbientLight,
    ambient_buffer: CpuBufferPool<fs::ty::Ambient_Data>,
    pipeline: Arc<GraphicsPipeline>,
    // Lumière directionnelle — gérée par LightController
    light: LightController,
    directional_buffer: CpuBufferPool<fs::ty::Directional_Light_Data>,
    // Skybox
    sky_settings: SkySettings,
    sky_pipeline: Arc<GraphicsPipeline>,
    sky_buffer: CpuBufferPool<sky_fs2::ty::Sky_Data>,
    // Billboard soleil
    sun_pipeline: Arc<GraphicsPipeline>,
    sun_data_buffer: CpuBufferPool<sun_vs::ty::Sun_Data>,
    sun_color_buffer: CpuBufferPool<sun_fs::ty::Sun_Color>,
    // Infrastructure
    queue: Arc<Queue>,
    device: Arc<Device>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,
}

impl Vulkmini {
    pub fn init() -> Self {
        let mut mvp = MVP::new();
        let camera = Camera::new();
        mvp.view = camera.view_matrix();
        mvp.model = translate(&identity(), &vec3(0.0, 0.0, -5.0));

        let ambient_light = AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.2,
        };

        // La lumière est contrôlée par LightController
        let light = LightController::new(
            vec3(-4.0, -4.0, 0.0),
            vec3(1.0, 1.0, 1.0),
        );

        let instance = {
            let library = VulkanLibrary::new().unwrap();
            let extensions = vulkano_win::required_extensions(&library);
            Instance::new(
                library,
                InstanceCreateInfo {
                    enabled_extensions: extensions,
                    enumerate_portability: true,
                    max_api_version: Some(Version::V1_1),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.graphics
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("No suitable physical device found");

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let caps = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let usage = caps.supported_usage_flags;
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            let image_extent: [u32; 2] = window.inner_size().into();
            let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
            mvp.projection = perspective(aspect_ratio, half_pi(), 0.01, 100.0);

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format,
                    image_extent,
                    image_usage: usage,
                    composite_alpha: alpha,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        // ── Shaders ──────────────────────────────────────────────────────────
        let vs_shader = vs::load(device.clone()).unwrap();
        let fs_shader = fs::load(device.clone()).unwrap();
        let sky_vs_shader = sky_vs2::load(device.clone()).unwrap();
        let sky_fs_shader = sky_fs2::load(device.clone()).unwrap();
        let sun_vs_shader = sun_vs::load(device.clone()).unwrap();
        let sun_fs_shader = sun_fs::load(device.clone()).unwrap();

        // ── Render pass ──────────────────────────────────────────────────────
        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )
        .unwrap();

        // ── Pipeline mesh principal ───────────────────────────────────────────
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();

        // ── Pipeline skybox ───────────────────────────────────────────────────
        // Pas de vertex buffer, pas de depth write (la skybox est toujours derrière).
        // On désactive le depth test pour la passe skybox en utilisant depth = 0.9999
        // émis dans le vertex shader (voir sky_vs2).
        // On utilise CullMode::None car il n'y a pas de géométrie réelle.
        use vulkano::pipeline::graphics::depth_stencil::{
            CompareOp, DepthState, DepthStencilState as DSS,
        };
        let sky_depth = DSS {
            depth: Some(DepthState {
                enable_dynamic: false,
                compare_op: vulkano::pipeline::StateMode::Fixed(CompareOp::LessOrEqual),
                write_enable: vulkano::pipeline::StateMode::Fixed(false), // pas d'écriture depth
            }),
            depth_bounds: None,
            stencil: None,
        };

        let sky_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new()) // pas de vertex buffer
            .vertex_shader(sky_vs_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(sky_fs_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(sky_depth)
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();

        // ── Pipeline billboard soleil ─────────────────────────────────────────
        // Pas de vertex buffer (4 sommets générés dans le vertex shader).
        // Pas de depth write pour éviter d'occulter le mesh, mais depth test
        // activé pour que le soleil soit caché derrière les objets proches.
        // Blending additif pour le halo lumineux.
        use vulkano::pipeline::graphics::color_blend::{
            AttachmentBlend, BlendFactor, BlendOp, ColorBlendState,
        };
        let sun_blend = ColorBlendState::new(1).blend(AttachmentBlend {
            color_op: BlendOp::Add,
            color_source: BlendFactor::SrcAlpha,
            color_destination: BlendFactor::OneMinusSrcAlpha,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::One,
            alpha_destination: BlendFactor::OneMinusSrcAlpha,
        });

        let sun_depth = DSS {
            depth: Some(DepthState {
                enable_dynamic: false,
                compare_op: vulkano::pipeline::StateMode::Fixed(CompareOp::Less),
                write_enable: vulkano::pipeline::StateMode::Fixed(false),
            }),
            depth_bounds: None,
            stencil: None,
        };

        use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology;
        let sun_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new())
            .vertex_shader(sun_vs_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip))
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(sun_fs_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(sun_depth)
            .color_blend_state(sun_blend)
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();

        // ── Buffer pools ─────────────────────────────────────────────────────
        let uniform_buffer: CpuBufferPool<vs::ty::MVP_Data> =
            CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let ambient_buffer: CpuBufferPool<fs::ty::Ambient_Data> =
            CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let directional_buffer: CpuBufferPool<fs::ty::Directional_Light_Data> =
            CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let sky_buffer: CpuBufferPool<sky_fs2::ty::Sky_Data> =
            CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let sun_data_buffer: CpuBufferPool<sun_vs::ty::Sun_Data> =
            CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let sun_color_buffer: CpuBufferPool<sun_fs::ty::Sun_Color> =
            CpuBufferPool::uniform_buffer(memory_allocator.clone());

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };
        let framebuffers = window_size_dependent_setup(
            &memory_allocator,
            &images,
            render_pass.clone(),
            &mut viewport,
        );

        Vulkmini {
            event_loop,
            surface,
            mvp,
            swapchain,
            framebuffers,
            memory_allocator,
            render_pass,
            viewport,
            camera,
            uniform_buffer,
            ambient_light,
            ambient_buffer,
            light,
            directional_buffer,
            sky_settings: SkySettings::default(),
            sky_pipeline,
            sky_buffer,
            sun_pipeline,
            sun_data_buffer,
            sun_color_buffer,
            pipeline,
            queue,
            device,
            descriptor_set_allocator,
            command_buffer_allocator,
        }
    }

    pub fn run(mut self, vertices: Vec<Vertex>) {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            vertices,
        )
        .unwrap();

        let mut recreate_swapchain = false;
        let rotation_start = Instant::now();
        let mut previous_frame_end =
            Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);

        self.event_loop.run(move |event, _, control_flow| match event {
            // ── Fermeture ────────────────────────────────────────────────────
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }

            // ── Clavier ──────────────────────────────────────────────────────
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if let Some(keycode) = input.virtual_keycode {
                    if input.state == ElementState::Pressed {
                        match keycode {
                            // Caméra
                            VirtualKeyCode::Up => self.camera.move_forward(0.1),
                            VirtualKeyCode::Down => self.camera.move_forward(-0.1),
                            VirtualKeyCode::Right => self.camera.strafe(0.1),
                            VirtualKeyCode::Left => self.camera.strafe(-0.1),
                            VirtualKeyCode::Q => self.camera.rotate_yaw(0.1),
                            VirtualKeyCode::D => self.camera.rotate_yaw(-0.1),
                            // Z / S → regarde en haut / en bas
                            VirtualKeyCode::S => self.camera.rotate_pitch(0.05),
                            VirtualKeyCode::Z => self.camera.rotate_pitch(-0.05),

                            // ── Lumière directionnelle ────────────────────
                            // O / L  → déplace sur X (gauche/droite)
                            VirtualKeyCode::O => self.light.move_x(-0.3),
                            VirtualKeyCode::L => self.light.move_x(0.3),
                            // K / M  → déplace sur Y (haut/bas)
                            VirtualKeyCode::K => self.light.move_y(0.3),
                            VirtualKeyCode::M => self.light.move_y(-0.3),
                            // N / J  → déplace sur Z (avant/arrière)
                            VirtualKeyCode::N => self.light.move_z(-0.3),
                            VirtualKeyCode::J => self.light.move_z(0.3),
                            // I / P  → intensité
                            VirtualKeyCode::I => self.light.change_intensity(0.1),
                            VirtualKeyCode::P => self.light.change_intensity(-0.1),

                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                            _ => {}
                        }
                    }
                }
            }

            // ── Rendu ────────────────────────────────────────────────────────
            Event::RedrawEventsCleared => {
                previous_frame_end
                    .as_mut()
                    .take()
                    .unwrap()
                    .cleanup_finished();

                if recreate_swapchain {
                    let window = self
                        .surface
                        .object()
                        .unwrap()
                        .downcast_ref::<Window>()
                        .unwrap();
                    let image_extent: [u32; 2] = window.inner_size().into();
                    let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
                    self.mvp.projection = perspective(aspect_ratio, half_pi(), 0.01, 100.0);

                    let (new_swapchain, new_images) =
                        match self.swapchain.recreate(SwapchainCreateInfo {
                            image_extent,
                            ..self.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    self.swapchain = new_swapchain;
                    self.framebuffers = window_size_dependent_setup(
                        &self.memory_allocator,
                        &new_images,
                        self.render_pass.clone(),
                        &mut self.viewport,
                    );
                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                // clear noir — la skybox remplacera ce fond de toute façon
                let clear_values = vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())];

                // ── Uniform mesh ──────────────────────────────────────────────
                // On calcule la vue une seule fois et on la partage entre mesh et soleil
                let view_matrix = self.camera.view_matrix();

                let uniform_subbuffer = {
                    let elapsed = rotation_start.elapsed().as_secs() as f64
                        + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                    let _elapsed_as_radians = elapsed * pi::<f64>() / 180.0;

                    let mut model: TMat4<f32> =
                        rotate_normalized_axis(&identity(), 1.0, &vec3(0.0, 0.0, 1.0));
                    model = rotate_normalized_axis(&model, 1.0, &vec3(0.0, 1.0, 0.0));
                    model = rotate_normalized_axis(&model, 1.0, &vec3(1.0, 0.0, 0.0));
                    model = self.mvp.model * model;

                    let uniform_data = vs::ty::MVP_Data {
                        model: model.into(),
                        view: view_matrix.into(),
                        projection: self.mvp.projection.into(),
                    };
                    self.uniform_buffer.from_data(uniform_data).unwrap()
                };

                // ── Uniform ambient ───────────────────────────────────────────
                let ambient_subbuffer = {
                    let uniform_data = fs::ty::Ambient_Data {
                        color: self.ambient_light.color.into(),
                        intensity: self.ambient_light.intensity.into(),
                    };
                    self.ambient_buffer.from_data(uniform_data).unwrap()
                };

                // ── Uniform lumière directionnelle ────────────────────────────
                let directional_subbuffer = {
                    let uniform_data = fs::ty::Directional_Light_Data {
                        position: self.light.position_vec4().into(),
                        color: self.light.effective_color().into(),
                    };
                    self.directional_buffer.from_data(uniform_data).unwrap()
                };

                // ── Descriptor set mesh ───────────────────────────────────────
                let mesh_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
                let mesh_set = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    mesh_layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, uniform_subbuffer),
                        WriteDescriptorSet::buffer(1, ambient_subbuffer),
                        WriteDescriptorSet::buffer(2, directional_subbuffer),
                    ],
                )
                .unwrap();

                // ── Uniform skybox ────────────────────────────────────────────
                // On calcule (projection * view)^-1 pour reconstruire les rays en world space
                let sky_subbuffer = {
                    let s = &self.sky_settings;
                    let proj_view = self.mvp.projection * view_matrix;
                    let inv_proj_view = nalgebra_glm::inverse(&proj_view);
                    let uniform_data = sky_fs2::ty::Sky_Data {
                        sky_color: s.sky_color.into(),
                        sharpness: s.sharpness,
                        ground_color: s.ground_color.into(),
                        _pad: 0.0,
                        inv_proj_view: inv_proj_view.into(),
                    };
                    self.sky_buffer.from_data(uniform_data).unwrap()
                };

                // ── Descriptor set skybox ─────────────────────────────────────
                let sky_layout = self.sky_pipeline.layout().set_layouts().get(0).unwrap();
                let sky_set = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    sky_layout.clone(),
                    [WriteDescriptorSet::buffer(0, sky_subbuffer)],
                )
                .unwrap();

                // ── Uniform billboard soleil ──────────────────────────────────
                let sun_data_sub = {
                    let p = self.light.position;
                    let data = sun_vs::ty::Sun_Data {
                        world_pos: [p.x, p.y, p.z],
                        size: 0.6,   // demi-taille en unités world — ajuste selon la scène
                        view: view_matrix.into(),
                        projection: self.mvp.projection.into(),
                    };
                    self.sun_data_buffer.from_data(data).unwrap()
                };
                let sun_color_sub = {
                    let ec = self.light.effective_color();
                    // Cœur : blanc chaud, halo : couleur effective de la lumière
                    let data = sun_fs::ty::Sun_Color {
                        core_color: [1.0, 0.97, 0.88],
                        _pad0: 0.0,
                        halo_color: ec,
                        _pad1: 0.0,
                    };
                    self.sun_color_buffer.from_data(data).unwrap()
                };

                let sun_layout = self.sun_pipeline.layout().set_layouts().get(0).unwrap();
                let sun_set = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    sun_layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, sun_data_sub),
                        WriteDescriptorSet::buffer(1, sun_color_sub),
                    ],
                )
                .unwrap();

                // ── Construction du command buffer ────────────────────────────
                let mut cmd = AutoCommandBufferBuilder::primary(
                    &self.command_buffer_allocator,
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                cmd.begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values,
                        ..RenderPassBeginInfo::framebuffer(
                            self.framebuffers[image_index as usize].clone(),
                        )
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .set_viewport(0, [self.viewport.clone()])
                // ── Passe 1 : skybox (triangle plein écran, depth=0.9999) ──────
                .bind_pipeline_graphics(self.sky_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.sky_pipeline.layout().clone(),
                    0,
                    sky_set,
                )
                // 3 sommets, pas de vertex buffer
                .draw(3, 1, 0, 0)
                .unwrap()
                // ── Passe 2 : mesh ────────────────────────────────────────────
                .bind_pipeline_graphics(self.pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    mesh_set,
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                // ── Passe 3 : billboard soleil ────────────────────────────────
                // Dessiné après le mesh pour que le blending fonctionne correctement.
                .bind_pipeline_graphics(self.sun_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.sun_pipeline.layout().clone(),
                    0,
                    sun_set,
                )
                // 4 sommets en triangle strip = quad
                .draw(4, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

                let command_buffer = cmd.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            self.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(Box::new(future) as Box<_>);
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end =
                            Some(Box::new(sync::now(self.device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end =
                            Some(Box::new(sync::now(self.device.clone())) as Box<_>);
                    }
                }
            }
            _ => (),
        });
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn window_size_dependent_setup(
    allocator: &StandardMemoryAllocator,
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(allocator, dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}