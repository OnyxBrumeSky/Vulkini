// Vulkano 0.33 : plus de `types_meta`, plus de `bytemuck` dans le shader!
// Les structs générées implémentent BufferContents automatiquement.
// On accède aux structs directement via `vs::MVP_Data` (plus de `vs::ty::MVP_Data`).

pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 450
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 normal;
                layout(location = 2) in vec3 color;

                layout(location = 0) out vec3 out_color;
                layout(location = 1) out vec3 out_normal;
                layout(location = 2) out vec3 frag_pos;

                layout(set = 0, binding = 0) uniform MVP_Data {
                    mat4 model;
                    mat4 view;
                    mat4 projection;
                } uniforms;

                void main() {
                    mat4 worldview = uniforms.view * uniforms.model;
                    gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
                    out_color = color;
                    out_normal = mat3(uniforms.model) * normal;
                    frag_pos = vec3(uniforms.model * vec4(position, 1.0));
                }
            ",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }

    
pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
                #version 450
                layout(location = 0) in vec3 in_color;
                layout(location = 1) in vec3 in_normal;
                layout(location = 2) in vec3 frag_pos;

                layout(location = 0) out vec4 f_color;

                layout(set = 0, binding = 1) uniform Ambient_Data {
                    vec3 color;
                    float intensity;
                } ambient;

                layout(set = 0, binding = 2) uniform Directional_Light_Data {
                    vec4 position;
                    vec3 color;
                } directional;

                void main() {
                    vec3 ambient_color = ambient.intensity * ambient.color;
                    vec3 light_direction = normalize(directional.position.xyz - frag_pos);
                    float directional_intensity = max(dot(in_normal, light_direction), 0.0);
                    vec3 directional_color = directional_intensity * directional.color;
                    vec3 combined_color = (ambient_color + directional_color) * in_color;
                    f_color = vec4(combined_color, 1.0);
                }
            ",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            }
        }
    }