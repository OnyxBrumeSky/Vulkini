// src/renderer/mod.rs
pub mod mesh;
pub mod skybox;
pub mod sun;

use vulkano::image::ImageAccess;
use std::sync::Arc;
use std::time::Instant;

use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::viewport::Viewport;
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

use nalgebra_glm::{half_pi, identity, perspective, translate, vec3, TMat4};

use crate::camera::{Camera, LightController};
use crate::input::InputState;
use crate::scene_object::{AmbientLight, SceneObject, Vertex, MVP};
use crate::texture::GpuTexture;

use self::mesh::MeshRenderer;
use self::skybox::{SkyboxRenderer, SkySettings};
use self::sun::SunRenderer;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};

pub struct Vulkmini {
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
    ambient_light: AmbientLight,
    light: LightController,
    sky_settings: SkySettings,
    white_texture: Arc<GpuTexture>,
    queue: Arc<Queue>,
    device: Arc<Device>,
    
    // Allocateurs d'infrastructure
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,

    // Sous-renderers modulaires
    skybox_renderer: SkyboxRenderer,
    mesh_renderer: MeshRenderer,
    sun_renderer: SunRenderer,
}

impl Vulkmini {
    pub fn init(event_loop: &EventLoop<()>) -> Self {
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

        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
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

        let white_texture = GpuTexture::create_white_fallback(
            &memory_allocator,
            &command_buffer_allocator,
            queue.clone(),
            device.clone(),
        );

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        
        // Initialisation de nos sous-renderers modulaires
        let skybox_renderer = SkyboxRenderer::new(device.clone(), memory_allocator.clone(), subpass.clone());
        let mesh_renderer = MeshRenderer::new(device.clone(), memory_allocator.clone(), subpass.clone());
        let sun_renderer = SunRenderer::new(device.clone(), memory_allocator.clone(), subpass);

        Vulkmini {
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
            ambient_light,
            light,
            sky_settings: SkySettings::default(),
            white_texture,
            queue,
            device,
            descriptor_set_allocator,
            command_buffer_allocator,
            skybox_renderer,
            mesh_renderer,
            sun_renderer,
        }
    }

    pub fn run(mut self, event_loop: EventLoop<()>, objects: Vec<SceneObject>) {
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

        event_loop.run(move |event, _, control_flow| match event {
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
                    self.input_state.update(keycode, input.state);
                    if keycode == VirtualKeyCode::Escape && is_pressed {
                        *control_flow = ControlFlow::Exit;
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
                let elapsed_time = rotation_start.elapsed().as_secs_f32();

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
                .set_viewport(0, [self.viewport.clone()]);

                // --- APPELS AUX SOUS-RENDERERS NETTOYÉS ET SÉPARÉS ---

                // 1. Rendu du fond (Skybox)
                self.skybox_renderer.draw(
                    &mut cmd,
                    &self.descriptor_set_allocator,
                    &self.mvp.projection,
                    &view_matrix,
                );

                // 2. Rendu des Meshs Opaques
                self.mesh_renderer.draw(
                    &mut cmd,
                    &self.descriptor_set_allocator,
                    &objects,
                    &vertex_buffers,
                    &self.mvp.projection,
                    &view_matrix,
                    self.ambient_light.color,
                    self.ambient_light.intensity,
                    &self.light,
                    &self.white_texture,
                    elapsed_time,
                );

                // 3. Rendu du Soleil
                self.sun_renderer.draw(
                    &mut cmd,
                    &self.descriptor_set_allocator,
                    &self.mvp.projection,
                    &view_matrix,
                    &self.light,
                );

                cmd.end_render_pass().unwrap();
                let command_buffer = builder_to_cmd(cmd);

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

    pub fn load_texture(&self, path: &str) -> Arc<GpuTexture> {
        GpuTexture::load_from_file(
            path,
            &self.memory_allocator,
            &self.command_buffer_allocator,
            self.queue.clone(),
            self.device.clone(),
        )
    }

}

// Helper pour finaliser l'écriture du Command Buffer
fn builder_to_cmd(builder: AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>) -> Arc<vulkano::command_buffer::PrimaryAutoCommandBuffer> {
    Arc::new(builder.build().unwrap())
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