use vulkini::vulkini::{Vulkmini, Vertex, SceneObject};
use nalgebra_glm::{translate, identity, vec3};
use std::sync::Arc;

fn main() {

    // UV mapping standard pour un cube (2 triangles par face)
    // Les coordonnées UV couvrent [0,1]x[0,1] sur chaque face.
    let cube_vertices = vec![
        // front face
        Vertex { position: [-1.0, -1.0,  1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [-1.0,  1.0,  1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [ 1.0,  1.0,  1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [-1.0, -1.0,  1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [ 1.0,  1.0,  1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [ 1.0, -1.0,  1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0] },
        // back face
        Vertex { position: [ 1.0, -1.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [ 1.0,  1.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [-1.0,  1.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [ 1.0, -1.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [-1.0,  1.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [-1.0, -1.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0] },
        // top face (y = -1)
        Vertex { position: [-1.0, -1.0,  1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [ 1.0, -1.0,  1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [ 1.0, -1.0, -1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0] },
        Vertex { position: [-1.0, -1.0,  1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [ 1.0, -1.0, -1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0] },
        Vertex { position: [-1.0, -1.0, -1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        // bottom face (y = +1)
        Vertex { position: [ 1.0,  1.0,  1.0], normal: [0.0, 1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [-1.0,  1.0,  1.0], normal: [0.0, 1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [-1.0,  1.0, -1.0], normal: [0.0, 1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [ 1.0,  1.0,  1.0], normal: [0.0, 1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [-1.0,  1.0, -1.0], normal: [0.0, 1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [ 1.0,  1.0, -1.0], normal: [0.0, 1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0] },
        // left face
        Vertex { position: [-1.0, -1.0, -1.0], normal: [-1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [-1.0,  1.0, -1.0], normal: [-1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [-1.0,  1.0,  1.0], normal: [-1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [-1.0, -1.0, -1.0], normal: [-1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [-1.0,  1.0,  1.0], normal: [-1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [-1.0, -1.0,  1.0], normal: [-1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0] },
        // right face
        Vertex { position: [ 1.0, -1.0,  1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [ 1.0,  1.0,  1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [ 1.0,  1.0, -1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [ 1.0, -1.0,  1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [ 1.0,  1.0, -1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [ 1.0, -1.0, -1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0] },
    ];

    // ─── Init du renderer ─────────────────────────────────────────────────────
    let mut app = Vulkmini::init();

    // ─── Chargement des textures ──────────────────────────────────────────────
    // Chargez vos PNG/JPG ici via app.load_texture("chemin/vers/fichier.png").
    // La méthode retourne un Arc<GpuTexture> à passer à with_texture().
    //
    // Exemple :
    //   let brique = Arc::new(app.load_texture("assets/brique.png"));
    //
    // Les objets sans texture utilisent automatiquement une texture blanche
    // (la couleur du vertex s'applique seule, comportement identique à avant).

    let ma_texture = app.load_texture("textures/cobblestone.png"); // ← adaptez le chemin

    // ─── Scène ────────────────────────────────────────────────────────────────
    let objects = vec![
        // Cube central avec texture, rotation lente
        SceneObject::new(
            cube_vertices.clone(),
            translate(&identity(), &vec3(0.0, 0.0, 0.0)),
        )
        .with_rotation(0.0)
        .with_texture(ma_texture.clone()),

        // Deuxième cube avec la même texture, rotation plus rapide
        SceneObject::new(
            cube_vertices.clone(),
            translate(&identity(), &vec3(4.0, 0.0, -5.0)),
        )
        .with_rotation(1.5)
        .with_texture(ma_texture.clone()),

        // Troisième cube sans texture → couleur unie (blanc × vertex color)
        SceneObject::new(
            cube_vertices.clone(),
            translate(&identity(), &vec3(-4.0, 1.0, -7.0)),
        ),
    ];

    app.run(objects);
}