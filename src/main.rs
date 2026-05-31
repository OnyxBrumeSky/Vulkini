// src/main.rs

// Déclarations des modules internes du moteur
mod camera;
mod shader;
mod ui;
mod ui_shader;
mod input;
mod texture;
mod scene_object;
mod renderer;

use nalgebra_glm::{identity, translate, vec3};
use winit::event_loop::EventLoop;

// Importation de tes structures depuis les modules nettoyés
use crate::scene_object::{SceneObject, Vertex};
use crate::renderer::Vulkmini;

fn main() {
    // 1. Initialisation de la boucle d'événements winit requise par Vulkano
    let event_loop = EventLoop::new();

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
    // On transmet désormais la référence de la boucle d'événements à l'initialisation
    let app = Vulkmini::init(&event_loop);

    // ─── Chargement des textures ──────────────────────────────────────────────
    let ma_texture = app.load_texture("textures/cobblestone.png");

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
            cube_vertices,
            translate(&identity(), &vec3(-4.0, 1.0, -7.0)),
        ),
    ];

    // ─── Exécution ────────────────────────────────────────────────────────────
    // La boucle d'événements est transmise en paramètre pour ouvrir et gérer la fenêtre
    app.run(event_loop, objects);
}