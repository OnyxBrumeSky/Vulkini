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
use vulkano::image::{
    AttachmentImage, ImageAccess, ImmutableImage, MipmapsCount, SwapchainImage,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
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
    half_pi, identity, perspective, rotate_normalized_axis, translate, vec3, TMat4,
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
    pub tex_coords: [f32; 2], // NOUVEAU : coordonnées UV pour les textures
}
vulkano::impl_vertex!(Vertex, position, normal, color, tex_coords);

// ─── Texture chargée sur le GPU ───────────────────────────────────────────────

/// Une texture prête à l'emploi pour être passée à SceneObject.
/// Créez-la via `Vulkmini::load_texture(path)` avant d'appeler `run()`.
pub struct GpuTexture {
    pub view: Arc<ImageView<ImmutableImage>>,
    pub sampler: Arc<Sampler>,
}

// ─── Objet de scène ───────────────────────────────────────────────────────────

pub struct SceneObject {
    pub vertices: Vec<Vertex>,
    pub base_transform: TMat4<f32>,
    pub rotation_speed: f32,
    /// Texture optionnelle. None → couleur unie (comportement original).
    pub texture: Option<Arc<GpuTexture>>,
}

impl SceneObject {
    pub fn new(vertices: Vec<Vertex>, base_transform: TMat4<f32>) -> Self {
        Self { vertices, base_transform, rotation_speed: 0.0, texture: None }
    }

    pub fn with_rotation(mut self, speed: f32) -> Self {
        self.rotation_speed = speed;
        self
    }

    /// Assigne une texture à cet objet.
    pub fn with_texture(mut self, tex: Arc<GpuTexture>) -> Self {
        self.texture = Some(tex);
        self
    }
}

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

// ─── Gestion du Clavier ──────────────────────────────────────────────────────

#[derive(Default)]
struct InputState {
    forward: bool,
    backward: bool,
    strafe_left: bool,
    strafe_right: bool,
    move_up: bool,
    move_down: bool,
    light_left: bool,
    light_right: bool,
    light_up: bool,
    light_down: bool,
    light_forward: bool,
    light_backward: bool,
    light_inc_intensity: bool,
    light_dec_intensity: bool,
}

// ─── Données skybox ──────────────────────────────────────────────────────────

pub struct SkySettings {
    pub sky_color: [f32; 3],
    pub ground_color: [f32; 3],
    pub sharpness: f32,
}

impl Default for SkySettings {
    fn default() -> Self {
        Self {
            sky_color: [0.18, 0.45, 0.82],
            ground_color: [0.08, 0.06, 0.05],
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
    input_state: InputState,
    last_frame: Instant,
    // Mesh
    uniform_buffer: CpuBufferPool<vs::ty::MVP_Data>,
    ambient_light: AmbientLight,
    ambient_buffer: CpuBufferPool<fs::ty::Ambient_Data>,
    pipeline: Arc<GraphicsPipeline>,
    // Lumière directionnelle
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
    // Texture blanche par défaut (fallback)
    white_texture: Arc<GpuTexture>,
    // Infrastructure
    queue: Arc<Queue>,
    device: Arc<Device>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,
}

impl Vulkmini {
    pub fn init() -> Self {
        let mut mvp = MVP::new();
        let mut camera = Camera::new();
        camera.update_target();
        mvp.view = camera.view_matrix();
        mvp.model = translate(&identity(), &vec3(0.0, 0.0, -5.0));

        let ambient_light = AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.2,
        };

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

        if let Some(window) = surface.object().unwrap().downcast_ref::<Window>() {
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
            window.set_cursor_visible(false);
        }

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

        let vs_shader = vs::load(device.clone()).unwrap();
        let fs_shader = fs::load(device.clone()).unwrap();
        let sky_vs_shader = sky_vs2::load(device.clone()).unwrap();
        let sky_fs_shader = sky_fs2::load(device.clone()).unwrap();
        let sun_vs_shader = sun_vs::load(device.clone()).unwrap();
        let sun_fs_shader = sun_fs::load(device.clone()).unwrap();

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

        use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState as DSS};
        let sky_depth = DSS {
            depth: Some(DepthState {
                enable_dynamic: false,
                compare_op: vulkano::pipeline::StateMode::Fixed(CompareOp::LessOrEqual),
                write_enable: vulkano::pipeline::StateMode::Fixed(false),
            }),
            depth_bounds: None,
            stencil: None,
        };

        let sky_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new())
            .vertex_shader(sky_vs_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(sky_fs_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(sky_depth)
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();

        use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState};
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

        let uniform_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let ambient_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let directional_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let sky_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let sun_data_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let sun_color_buffer = CpuBufferPool::uniform_buffer(memory_allocator.clone());

        // ── Texture blanche 1×1 (fallback pour les objets sans texture) ──────
        let white_texture = Self::load_white_texture(
            &memory_allocator,
            queue.clone(),
            &command_buffer_allocator,
            &device,
        );

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
            input_state: InputState::default(),
            last_frame: Instant::now(),
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
            white_texture,
            pipeline,
            queue,
            device,
            descriptor_set_allocator,
            command_buffer_allocator,
        }
    }

    // ── Chargement d'une texture depuis un fichier PNG/JPG ────────────────────
    //
    // Appelez cette méthode AVANT `run()` pour obtenir un Arc<GpuTexture>
    // à passer à SceneObject::with_texture().
    //
    // Exemple :
    //   let tex = app.load_texture("assets/brick.png");
    //   let obj = SceneObject::new(verts, transform).with_texture(tex);
    //
    pub fn load_texture(&self, path: &str) -> Arc<GpuTexture> {
        // Décodage du fichier image avec la crate `image`
        let img = image::open(path)
            .unwrap_or_else(|e| panic!("Impossible d'ouvrir la texture '{}' : {}", path, e))
            .into_rgba8();

        let (width, height) = img.dimensions();
        let rgba_data: Vec<u8> = img.into_raw();

        // Upload vers le GPU via un command buffer one-shot
        let mut upload_cmd = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let gpu_image = ImmutableImage::from_iter(
            &self.memory_allocator,
            rgba_data,
            vulkano::image::ImageDimensions::Dim2d { width, height, array_layers: 1 },
            MipmapsCount::One,
            Format::R8G8B8A8_UNORM,
            &mut upload_cmd,
        )
        .unwrap();

        // Exécution du command buffer de transfert
        let upload_cmd = upload_cmd.build().unwrap();
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), upload_cmd)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let view = ImageView::new_default(gpu_image).unwrap();
        let sampler = Sampler::new(
            self.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        Arc::new(GpuTexture { view, sampler })
    }

    // ── Texture blanche interne (1×1, utilisée quand pas de texture) ──────────
    fn load_white_texture(
        memory_allocator: &StandardMemoryAllocator,
        queue: Arc<Queue>,
        cmd_alloc: &StandardCommandBufferAllocator,
        device: &Arc<Device>,
    ) -> Arc<GpuTexture> {
        let pixels: Vec<u8> = vec![255, 255, 255, 255];

        let mut upload_cmd = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let gpu_image = ImmutableImage::from_iter(
            memory_allocator,
            pixels,
            vulkano::image::ImageDimensions::Dim2d { width: 1, height: 1, array_layers: 1 },
            MipmapsCount::One,
            Format::R8G8B8A8_UNORM,
            &mut upload_cmd,
        )
        .unwrap();

        let upload_cmd = upload_cmd.build().unwrap();
        sync::now(device.clone())
            .then_execute(queue.clone(), upload_cmd)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let view = ImageView::new_default(gpu_image).unwrap();
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        Arc::new(GpuTexture { view, sampler })
    }

    pub fn run(mut self, objects: Vec<SceneObject>) {
        let vertex_buffers: Vec<Arc<CpuAccessibleBuffer<[Vertex]>>> = objects
            .iter()
            .map(|obj| {
                CpuAccessibleBuffer::from_iter(
                    &self.memory_allocator,
                    BufferUsage { vertex_buffer: true, ..BufferUsage::empty() },
                    false,
                    obj.vertices.iter().cloned(),
                )
                .unwrap()
            })
            .collect();

        let mut recreate_swapchain = false;
        let rotation_start = Instant::now();
        let mut previous_frame_end =
            Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);

        self.event_loop.run(move |event, _, control_flow| match event {
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

            Event::DeviceEvent {
                event: winit::event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                let sensitivity = 0.0015;
                self.camera.rotate_mouse(delta.0 as f32, delta.1 as f32, sensitivity);
            }

            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if let Some(keycode) = input.virtual_keycode {
                    let is_pressed = input.state == ElementState::Pressed;
                    match keycode {
                        VirtualKeyCode::Up => self.input_state.forward = is_pressed,
                        VirtualKeyCode::Down => self.input_state.backward = is_pressed,
                        VirtualKeyCode::Right => self.input_state.strafe_right = is_pressed,
                        VirtualKeyCode::Left => self.input_state.strafe_left = is_pressed,
                        VirtualKeyCode::Space => self.input_state.move_down = is_pressed,
                        VirtualKeyCode::LShift | VirtualKeyCode::RShift => self.input_state.move_up = is_pressed,
                        VirtualKeyCode::O => self.input_state.light_left = is_pressed,
                        VirtualKeyCode::L => self.input_state.light_right = is_pressed,
                        VirtualKeyCode::K => self.input_state.light_up = is_pressed,
                        VirtualKeyCode::M => self.input_state.light_down = is_pressed,
                        VirtualKeyCode::N => self.input_state.light_forward = is_pressed,
                        VirtualKeyCode::J => self.input_state.light_backward = is_pressed,
                        VirtualKeyCode::I => self.input_state.light_inc_intensity = is_pressed,
                        VirtualKeyCode::P => self.input_state.light_dec_intensity = is_pressed,
                        VirtualKeyCode::Escape if is_pressed => *control_flow = ControlFlow::Exit,
                        _ => {}
                    }
                }
            }

            Event::RedrawEventsCleared => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                let speed = 4.0 * dt;
                let light_speed = 5.0 * dt;
                let light_intensity_speed = 2.0 * dt;

                if self.input_state.forward { self.camera.move_forward(speed); }
                if self.input_state.backward { self.camera.move_forward(-speed); }
                if self.input_state.strafe_right { self.camera.strafe(speed); }
                if self.input_state.strafe_left { self.camera.strafe(-speed); }
                if self.input_state.move_up { self.camera.move_up(speed); }
                if self.input_state.move_down { self.camera.move_up(-speed); }
                self.camera.update_target();

                if self.input_state.light_left { self.light.move_x(-light_speed); }
                if self.input_state.light_right { self.light.move_x(light_speed); }
                if self.input_state.light_up { self.light.move_y(light_speed); }
                if self.input_state.light_down { self.light.move_y(-light_speed); }
                if self.input_state.light_forward { self.light.move_z(-light_speed); }
                if self.input_state.light_backward { self.light.move_z(light_speed); }
                if self.input_state.light_inc_intensity { self.light.change_intensity(light_intensity_speed); }
                if self.input_state.light_dec_intensity { self.light.change_intensity(-light_intensity_speed); }

                previous_frame_end.as_mut().take().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let window = self.surface.object().unwrap().downcast_ref::<Window>().unwrap();
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

                if suboptimal { recreate_swapchain = true; }

                let clear_values = vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())];
                let view_matrix = self.camera.view_matrix();

                let ambient_subbuffer = {
                    let uniform_data = fs::ty::Ambient_Data {
                        color: self.ambient_light.color.into(),
                        intensity: self.ambient_light.intensity.into(),
                    };
                    self.ambient_buffer.from_data(uniform_data).unwrap()
                };

                let directional_subbuffer = {
                    let uniform_data = fs::ty::Directional_Light_Data {
                        position: self.light.position_vec4().into(),
                        color: self.light.effective_color().into(),
                    };
                    self.directional_buffer.from_data(uniform_data).unwrap()
                };

                let mesh_layout = self.pipeline.layout().set_layouts().get(0).unwrap().clone();

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

                let sky_layout = self.sky_pipeline.layout().set_layouts().get(0).unwrap();
                let sky_set = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    sky_layout.clone(),
                    [WriteDescriptorSet::buffer(0, sky_subbuffer)],
                )
                .unwrap();

                let sun_data_sub = {
                    let p = self.light.position;
                    let data = sun_vs::ty::Sun_Data {
                        world_pos: [p.x, p.y, p.z],
                        size: 0.6,
                        view: view_matrix.into(),
                        projection: self.mvp.projection.into(),
                    };
                    self.sun_data_buffer.from_data(data).unwrap()
                };
                let sun_color_sub = {
                    let ec = self.light.effective_color();
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
                // ── Skybox ──────────────────────────────────────────────────
                .set_viewport(0, [self.viewport.clone()])
                .bind_pipeline_graphics(self.sky_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.sky_pipeline.layout().clone(),
                    0,
                    sky_set,
                )
                .draw(3, 1, 0, 0)
                .unwrap()
                // ── Mesh pipeline ───────────────────────────────────────────
                .bind_pipeline_graphics(self.pipeline.clone());

                let elapsed = rotation_start.elapsed().as_secs_f32();
                for (obj, vbuf) in objects.iter().zip(vertex_buffers.iter()) {
                    let angle = elapsed * obj.rotation_speed;
                    let model = rotate_normalized_axis(&obj.base_transform, angle, &vec3(0.0, 1.0, 0.0));

                    let uniform_subbuffer = {
                        let uniform_data = vs::ty::MVP_Data {
                            model: model.into(),
                            view: view_matrix.into(),
                            projection: self.mvp.projection.into(),
                        };
                        self.uniform_buffer.from_data(uniform_data).unwrap()
                    };

                    // Texture : utilise celle de l'objet ou la blanche par défaut
                    let tex = obj.texture.as_ref().unwrap_or(&self.white_texture);

                    let mesh_set = PersistentDescriptorSet::new(
                        &self.descriptor_set_allocator,
                        mesh_layout.clone(),
                        [
                            WriteDescriptorSet::buffer(0, uniform_subbuffer),
                            WriteDescriptorSet::buffer(1, ambient_subbuffer.clone()),
                            WriteDescriptorSet::buffer(2, directional_subbuffer.clone()),
                            // binding 3 : sampler + image de la texture
                            WriteDescriptorSet::image_view_sampler(
                                3,
                                tex.view.clone(),
                                tex.sampler.clone(),
                            ),
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

                // ── Billboard soleil ────────────────────────────────────────
                cmd
                .bind_pipeline_graphics(self.sun_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.sun_pipeline.layout().clone(),
                    0,
                    sun_set,
                )
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