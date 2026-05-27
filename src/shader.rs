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

// ─── Skybox ───────────────────────────────────────────────────────────────────
// Un quad plein écran dessiné avec gl_VertexIndex (pas de vertex buffer).
// Le fragment shader produit un dégradé vertical : ciel en haut, sol en bas.
// On utilise un uniform séparé pour les couleurs afin de pouvoir les changer.

pub mod sky_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            // Génère un triangle couvrant tout l'écran sans vertex buffer
            // (technique du 'fullscreen triangle')
            void main() {
                // 3 sommets d'un triangle qui couvre [-1,1]x[-1,1]
                vec2 positions[3] = vec2[](
                    vec2(-1.0, -1.0),
                    vec2( 3.0, -1.0),
                    vec2(-1.0,  3.0)
                );
                gl_Position = vec4(positions[gl_VertexIndex], 0.9999, 1.0);
                // depth = 0.9999 pour être derrière tout le reste
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
                vec3 sky_color;      // couleur du ciel (haut)
                float _pad0;
                vec3 horizon_color;  // couleur à l'horizon
                float _pad1;
                vec3 ground_color;   // couleur du sol (bas)
                float _pad2;
                float horizon_sharpness; // contrôle la dureté de la ligne d'horizon
            } sky;

            void main() {
                // gl_FragCoord.y va de 0 (haut) à height (bas) en Vulkan
                // On utilise la position NDC reconstruite depuis gl_FragCoord
                // Pour un fullscreen triangle, y NDC = (gl_FragCoord.y / dFdy) * 2 - 1
                // Plus simple : on utilise directement la position clip reconstruite.
                // Comme depth=0.9999 et w=1, on peut approximer depuis gl_FragCoord.
                // On calcule juste t depuis la position y du fragment en NDC.
                float ndc_y = gl_FragCoord.y; // pixels, mais on veut 0..1
                // dFdy n'est pas disponible facilement ici ; on passe par gl_FragCoord
                // et on note que le triangle génère y_clip de -1 (bas) à 3 (haut).
                // On utilise une astuce : la dérivée partielle de y.
                float dy = dFdy(gl_FragCoord.y);  // ~1 ou -1 selon convention
                // Approximation du gradient vertical : on se base sur gl_FragCoord.y
                // normalisé dans [0,1] en utilisant fwidth pour être résolution-indépendant.
                // Méthode robuste : passer la position clip en out du vertex shader.
                // Pour garder le code simple on utilise un push constant indirect.
                // *** Méthode finale : on reconstruit depuis la position NDC ***
                // Le vertex shader émet des positions Y clip de -1 à 3.
                // La rasterisation interpole. On n'a pas de varying, mais on peut
                // utiliser gl_FragCoord combiné à la résolution.
                // Pour éviter un uniform supplémentaire, on utilise dFdx/dFdy
                // pour détecter si on est en haut ou en bas de l'écran :
                // si dy < 0, convention Vulkan (y flippé), sinon OpenGL.
                // Approche la plus simple : émettre t depuis le vertex shader.
                // → On refait le shader avec un varying.
                f_color = vec4(1.0, 0.0, 1.0, 1.0); // placeholder, voir sky_vs_v2
            }
        ",
    }
}

// Skybox consciente de la caméra.
// Le fragment shader reconstruit le ray de vue en world space via
// (projection * view)^-1, puis utilise ray.y pour distinguer
// ciel (y > 0) et sol (y < 0). L'horizon reste fixe peu importe le pitch.

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
                // Reconstruction du ray de vue en world space
                vec4 clip = vec4(ndc_pos, 1.0, 1.0);
                vec4 world = sky.inv_proj_view * clip;
                vec3 ray_dir = normalize(world.xyz / world.w);

                // ray_dir.y : +1=zenith, -1=nadir, 0=horizon
                // On nie ray_dir.y car Vulkan a l'axe Y clip inversé par rapport à OpenGL
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
// Un quad de 4 sommets centré sur la position de la lumière, toujours face
// à la caméra (billboard sphérique). Le vertex shader reconstruit les coins
// en world space depuis les axes right/up de la matrice de vue.
// Le fragment shader dessine un disque avec halo progressif.

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

                // Axes right et up extraits de la matrice de vue (billboard sphérique)
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