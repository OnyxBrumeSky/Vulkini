// pub mod ui_vs {
//     vulkano_shaders::shader! {
//         ty: "vertex",
//         src: "
//             #version 450
//             layout(location = 0) in vec2 position;
//             layout(location = 1) in vec2 tex_coords;
//             layout(location = 2) in vec4 color;

//             layout(location = 0) out vec2 out_tex_coords;
//             layout(location = 1) out vec4 out_color;

//             // Push constants pour éviter un buffer uniform lourd juste pour la taille écran
//             layout(push_constant) uniform PushConstants {
//                 vec2 screen_size;
//             } pcs;

//             void main() {
//                 // Conversion de l'espace pixel [0, W]x[0, H] vers l'espace NDC de Vulkan [-1, 1]
//                 // Top-left = (-1, -1), Bottom-right = (1, 1)
//                 float nx = (position.x / pcs.screen_size.x) * 2.0 - 1.0;
//                 float ny = (position.y / pcs.screen_size.y) * 2.0 - 1.0;

//                 gl_Position = vec4(nx, ny, 0.0, 1.0);
//                 out_tex_coords = tex_coords;
//                 out_color = color;
//             }
//         ",
//         types_meta: {
//             use bytemuck::{Pod, Zeroable};
//             #[derive(Clone, Copy, Zeroable, Pod)]
//         },
//     }
// }

// pub mod ui_fs {
//     vulkano_shaders::shader! {
//         ty: "fragment",
//         src: "
//             #version 450
//             layout(location = 0) in vec2 in_tex_coords;
//             layout(location = 1) in vec4 in_color;

//             layout(location = 0) out vec4 f_color;

//             layout(set = 0, binding = 0) uniform sampler2D font_atlas;

//             void main() {
//                 // L'atlas étant au format R8_UNORM, la texture renvoie la valeur dans la composante .r
//                 float alpha = texture(font_atlas, in_tex_coords).r;
                
//                 // Si l'alpha est trop bas, on peut discard ou laisser le blending faire
//                 f_color = vec4(in_color.rgb, in_color.a * alpha);
//             }
//         "
//     }
// }