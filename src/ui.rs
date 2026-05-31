// use std::collections::HashMap;
// use std::sync::Arc;
// use bytemuck::{Pod, Zeroable};
// use fontdue::{Font, FontSettings};
// use vulkano::image::view::ImageView;
// use vulkano::image::{ImmutableImage, ImageDimensions, MipmapsCount};
// use vulkano::format::Format;
// use vulkano::sampler::{Sampler, SamplerCreateInfo, Filter, SamplerAddressMode};
// use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
// use vulkano::sync::{self, GpuFuture};
// use vulkano::device::Device;
// use vulkano::memory::allocator::StandardMemoryAllocator;
// use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;

// #[repr(C)]
// #[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
// pub struct UiVertex {
//     pub position: [f32; 2],
//     pub tex_coords: [f32; 2],
//     pub color: [f32; 4],
// }
// vulkano::impl_vertex!(UiVertex, position, tex_coords, color);

// // Stocke les informations géométriques d'un glyphe dans l'atlas
// struct GlyphInfo {
//     uv_min: [f32; 2],
//     uv_max: [f32; 2],
//     width: f32,
//     height: f32,
//     bearing_x: f32,
//     bearing_y: f32,
//     advance: f32,
// }

// pub struct FontUiSystem {
//     font: Font,
//     glyphs: HashMap<char, GlyphInfo>,
//     pub atlas_view: Arc<ImageView<ImmutableImage>>,
//     pub atlas_sampler: Arc<Sampler>,
// }

// impl FontUiSystem {
//     pub fn new(
//         font_bytes: &[u8],
//         font_size: f32,
//         device: Arc<Device>,
//         memory_allocator: &StandardMemoryAllocator,
//         cmd_alloc: &StandardCommandBufferAllocator,
//         queue: Arc<vulkano::device::Queue>,
//     ) -> Self {
//         let font = Font::from_bytes(font_bytes, FontSettings::default()).unwrap();
        
//         // Configuration de l'atlas
//         let atlas_dim = 512;
//         let mut atlas_data = vec![0u8; atlas_dim * atlas_dim];
//         let mut glyphs = HashMap::new();

//         // Remplissage de l'atlas simple (Algorithme en étagère / Shelf packing)
//         let mut current_x = 0;
//         let mut current_y = 0;
//         let mut row_h = 0;

//         // On pack les caractères ASCII imprimables standard
//         for c in ' '..='~' {
//             let (metrics, bitmap) = font.rasterize(c, font_size);
            
//             if current_x + metrics.width > atlas_dim {
//                 current_x = 0;
//                 current_y += row_h + 2;
//                 row_h = 0;
//             }

//             if current_y + metrics.height > atlas_dim {
//                 panic!("L'atlas de texture de l'UI est trop petit ! Augmentez la taille (ex: 1024).");
//             }

//             // Copie du bitmap rasterisé dans notre atlas global
//             for y in 0..metrics.height {
//                 for x in 0..metrics.width {
//                     let src_idx = y * metrics.width + x;
//                     let dst_idx = (current_y + y) * atlas_dim + (current_x + x);
//                     atlas_data[dst_idx] = bitmap[src_idx];
//                 }
//             }

//             // Calcul des coordonnées UV normalisées [0.0, 1.0]
//             let uv_min = [current_x as f32 / atlas_dim as f32, current_y as f32 / atlas_dim as f32];
//             let uv_max = [
//                 (current_x + metrics.width) as f32 / atlas_dim as f32,
//                 (current_y + metrics.height) as f32 / atlas_dim as f32,
//             ];

//             glyphs.insert(c, GlyphInfo {
//                 uv_min,
//                 uv_max,
//                 width: metrics.width as f32,
//                 height: metrics.height as f32,
//                 bearing_x: metrics.xmin as f32,
//                 bearing_y: metrics.ymin as f32,
//                 advance: metrics.advance_width,
//             });

//             current_x += metrics.width + 2;
//             row_h = row_h.max(metrics.height);
//         }

//         // Création de l'Image Vulkano au format R8_UNORM (parfait pour le texte/niveaux de gris)
//         let mut upload_cmd = AutoCommandBufferBuilder::primary(
//             cmd_alloc,
//             queue.queue_family_index(),
//             CommandBufferUsage::OneTimeSubmit,
//         ).unwrap();

//         let gpu_image = ImmutableImage::from_iter(
//             memory_allocator,
//             atlas_data,
//             ImageDimensions::Dim2d { width: atlas_dim as u32, height: atlas_dim as u32, array_layers: 1 },
//             MipmapsCount::One,
//             Format::R8_UNORM, // Format léger canal Alpha unique
//             &mut upload_cmd,
//         ).unwrap();

//         let upload_cmd = upload_cmd.build().unwrap();
//         sync::now(device.clone())
//             .then_execute(queue.clone(), upload_cmd).unwrap()
//             .then_signal_fence_and_flush().unwrap()
//             .wait(None).unwrap();

//         let atlas_view = ImageView::new_default(gpu_image).unwrap();
//         let atlas_sampler = Sampler::new(
//             device,
//             SamplerCreateInfo {
//                 mag_filter: Filter::Linear,
//                 min_filter: Filter::Linear,
//                 address_mode: [SamplerAddressMode::ClampToEdge; 3],
//                 ..Default::default()
//             },
//         ).unwrap();

//         Self { font, glyphs, atlas_view, atlas_sampler }
//     }

//     /// Génère une liste de vertices (Quads) pour une chaîne de caractères donnée
//     pub fn generate_text_vertices(&self, text: &str, mut x: f32, y: f32, color: [f32; 4]) -> Vec<UiVertex> {
//         let mut vertices = Vec::new();

//         for c in text.chars() {
//             if let Some(info) = self.glyphs.get(&c) {
//                 // Calcul du positionnement précis (basé sur la ligne de base du texte)
//                 let x_pos = x + info.bearing_x;
//                 // Fontdue utilise un axe Y vers le haut pour la bearing_y, on l'inverse pour les coordonnées écran standard
//                 let y_pos = y - info.bearing_y; 

//                 let w = info.width;
//                 let h = info.height;

//                 // Construction du Quad (2 triangles = 6 vertices)
//                 let v1 = UiVertex { position: [x_pos, y_pos],         tex_coords: [info.uv_min[0], info.uv_min[1]], color };
//                 let v2 = UiVertex { position: [x_pos + w, y_pos],     tex_coords: [info.uv_max[0], info.uv_min[1]], color };
//                 let v3 = UiVertex { position: [x_pos, y_pos + h],     tex_coords: [info.uv_min[0], info.uv_max[1]], color };
//                 let v4 = UiVertex { position: [x_pos + w, y_pos + h], tex_coords: [info.uv_max[0], info.uv_max[1]], color };

//                 // Triangle 1
//                 vertices.push(v1);
//                 vertices.push(v2);
//                 vertices.push(v3);
//                 // Triangle 2
//                 vertices.push(v2);
//                 vertices.push(v4);
//                 vertices.push(v3);

//                 // Avancement horizontal
//                 x += info.advance;
//             }
//         }
//         vertices
//     }
// }