// shader.rs
// Contient les shaders du mesh principal + les shaders de la skybox + le billboard soleil.

// ─── Mesh principal ───────────────────────────────────────────────────────────

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec3 color;
            layout(location = 3) in vec2 tex_coords; // NOUVEAU

            layout(location = 0) out vec3 out_color;
            layout(location = 1) out vec3 out_normal;
            layout(location = 2) out vec3 frag_pos;
            layout(location = 3) out vec2 out_tex_coords; // NOUVEAU

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
                out_tex_coords = tex_coords; // NOUVEAU
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
            layout(location = 3) in vec2 in_tex_coords; // NOUVEAU

            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 1) uniform Ambient_Data {
                vec3 color;
                float intensity;
            } ambient;

            layout(set = 0, binding = 2) uniform Directional_Light_Data {
                vec4 position;
                vec3 color;
            } directional;

            // NOUVEAU : sampler de texture (binding 3)
            layout(set = 0, binding = 3) uniform sampler2D tex;

            void main() {
                vec3 ambient_color = ambient.intensity * ambient.color;
                vec3 light_direction = normalize(directional.position.xyz - frag_pos);
                float directional_intensity = max(dot(in_normal, light_direction), 0.0);
                vec3 directional_color = directional_intensity * directional.color;

                // Echantillonnage de la texture, multiplié par la couleur du vertex
                vec4 tex_sample = texture(tex, in_tex_coords);
                vec3 base_color = tex_sample.rgb * in_color;

                vec3 combined_color = (ambient_color + directional_color) * base_color;
                f_color = vec4(combined_color, tex_sample.a);
            }
        ",
        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy, Zeroable, Pod)]
        }
    }
}

// ─── Skybox ───────────────────────────────────────────────────────────────────

pub mod sky_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            void main() {
                vec2 positions[3] = vec2[](
                    vec2(-1.0, -1.0),
                    vec2( 3.0, -1.0),
                    vec2(-1.0,  3.0)
                );
                gl_Position = vec4(positions[gl_VertexIndex], 0.9999, 1.0);
            }
        ",
    }
}

pub mod sky_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) out vec4 f_color;
            layout(set = 0, binding = 0) uniform Sky_Data {
                vec3 sky_color;
                float _pad0;
                vec3 horizon_color;
                float _pad1;
                vec3 ground_color;
                float _pad2;
                float horizon_sharpness;
            } sky;
            void main() {
                f_color = vec4(1.0, 0.0, 1.0, 1.0);
            }
        ",
    }
}

pub mod sky_vs2 {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            layout(location = 0) out vec2 ndc_pos;

            void main() {
                vec2 positions[3] = vec2[](
                    vec2(-1.0,  1.0),
                    vec2( 3.0,  1.0),
                    vec2(-1.0, -3.0)
                );
                vec2 pos = positions[gl_VertexIndex];
                ndc_pos = pos;
                gl_Position = vec4(pos, 0.9999, 1.0);
            }
        ",
    }
}

pub mod sky_fs2 {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) in vec2 ndc_pos;
            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 0) uniform Sky_Data {
                vec3  sky_color;
                float sharpness;
                vec3  ground_color;
                float _pad;
                mat4  inv_proj_view;
            } sky;

            void main() {
                vec4 clip = vec4(ndc_pos, 1.0, 1.0);
                vec4 world = sky.inv_proj_view * clip;
                vec3 ray_dir = normalize(world.xyz / world.w);

                float hw = 1.0 / sky.sharpness;
                float blend = smoothstep(-hw, hw, -ray_dir.y);

                vec3 color = mix(sky.ground_color, sky.sky_color, blend);
                f_color = vec4(color, 1.0);
            }
        ",
        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy, Zeroable, Pod)]
        }
    }
}


// ─── Billboard soleil ─────────────────────────────────────────────────────────

pub mod sun_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            layout(location = 0) out vec2 uv;

            layout(set = 0, binding = 0) uniform Sun_Data {
                vec3 world_pos;
                float size;
                mat4 view;
                mat4 projection;
            } sun;

            void main() {
                vec2 corners[4] = vec2[](
                    vec2(-1.0, -1.0),
                    vec2( 1.0, -1.0),
                    vec2(-1.0,  1.0),
                    vec2( 1.0,  1.0)
                );
                vec2 corner = corners[gl_VertexIndex];
                uv = corner;

                vec3 cam_right = vec3(sun.view[0][0], sun.view[1][0], sun.view[2][0]);
                vec3 cam_up    = vec3(sun.view[0][1], sun.view[1][1], sun.view[2][1]);

                vec3 world_vertex = sun.world_pos
                    + cam_right * corner.x * sun.size
                    + cam_up    * corner.y * sun.size;

                gl_Position = sun.projection * sun.view * vec4(world_vertex, 1.0);
            }
        ",
        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy, Zeroable, Pod)]
        }
    }
}

pub mod sun_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) in vec2 uv;
            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 0) uniform Sun_Data {
                vec3 world_pos;
                float size;
                mat4 view;
                mat4 projection;
            } sun;

            layout(set = 0, binding = 1) uniform Sun_Color {
                vec3 core_color;
                float _pad0;
                vec3 halo_color;
                float _pad1;
            } colors;

            void main() {
                float dist = length(uv);
                if (dist > 1.0) discard;

                float core = 1.0 - smoothstep(0.0, 0.25, dist);
                float halo = pow(1.0 - smoothstep(0.15, 1.0, dist), 2.5);

                vec3 color = mix(colors.halo_color, colors.core_color, core);
                float alpha = max(core, halo * 0.85);

                f_color = vec4(color, alpha);
            }
        ",
        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy, Zeroable, Pod)]
        }
    }
}