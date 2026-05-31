// src/texture.rs
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{ImmutableImage, MipmapsCount};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::sync::{self, GpuFuture};

pub struct GpuTexture {
    pub view: Arc<ImageView<ImmutableImage>>,
    pub sampler: Arc<Sampler>,
}

impl GpuTexture {
    /// Charge une image (PNG/JPG) depuis un chemin donné vers la mémoire GPU.
    pub fn load_from_file(
        path: &str,
        allocator: &StandardMemoryAllocator,
        cmd_alloc: &StandardCommandBufferAllocator,
        queue: Arc<Queue>,
        device: Arc<Device>,
    ) -> Arc<Self> {
        let img = image::open(path)
            .unwrap_or_else(|e| panic!("Impossible d'ouvrir la texture '{}' : {}", path, e))
            .into_rgba8();

        let (width, height) = img.dimensions();
        let rgba_data: Vec<u8> = img.into_raw();

        let mut upload_cmd = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let gpu_image = ImmutableImage::from_iter(
            allocator,
            rgba_data,
            vulkano::image::ImageDimensions::Dim2d { width, height, array_layers: 1 },
            MipmapsCount::One,
            Format::R8G8B8A8_UNORM,
            &mut upload_cmd,
        )
        .unwrap();

        let upload_cmd = upload_cmd.build().unwrap();
        sync::now(device.clone())
            .then_execute(queue, upload_cmd)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let view = ImageView::new_default(gpu_image).unwrap();
        let sampler = Sampler::new(
            device,
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

    /// Crée une texture blanche interne 1x1 servant de fallback pour les objets sans texture.
    pub fn create_white_fallback(
        allocator: &StandardMemoryAllocator,
        cmd_alloc: &StandardCommandBufferAllocator,
        queue: Arc<Queue>,
        device: Arc<Device>,
    ) -> Arc<Self> {
        let pixels: Vec<u8> = vec![255, 255, 255, 255];

        let mut upload_cmd = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let gpu_image = ImmutableImage::from_iter(
            allocator,
            pixels,
            vulkano::image::ImageDimensions::Dim2d { width: 1, height: 1, array_layers: 1 },
            MipmapsCount::One,
            Format::R8G8B8A8_UNORM,
            &mut upload_cmd,
        )
        .unwrap();

        let upload_cmd = upload_cmd.build().unwrap();
        sync::now(device.clone())
            .then_execute(queue, upload_cmd)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let view = ImageView::new_default(gpu_image).unwrap();
        let sampler = Sampler::new(
            device,
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
}