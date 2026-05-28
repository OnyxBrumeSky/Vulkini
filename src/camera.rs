// camera.rs
use nalgebra_glm::{Vec3, vec3, look_at, normalize, cross, rotation};
use nalgebra_glm::TMat4;

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    // NOUVEAU : Suivi des angles pour la gestion fluide à la souris
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            eye: vec3(0.0, 0.0, 0.5),
            target: vec3(0.0, 0.0, 0.0),
            up: vec3(0.0, 1.0, 0.0),
            // NOUVEAU : Initialisation à -90 degrés sur l'axe Y pour regarder vers -Z par défaut
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        }
    }

    pub fn view_matrix(&self) -> TMat4<f32> {
        look_at(&self.eye, &self.target, &self.up)
    }

    // NOUVEAU : Recalcule le vecteur cible (target) basé sur les angles yaw/pitch et la position de l'œil
    pub fn update_target(&mut self) {
        let front = vec3(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );
        self.target = self.eye + normalize(&front);
    }

    pub fn move_forward(&mut self, amount: f32) {
        let dir = normalize(&(self.target - self.eye));
        self.eye += dir * amount;
        self.target += dir * amount;
    }

    pub fn strafe(&mut self, amount: f32) {
        let dir = normalize(&(self.target - self.eye));
        let right = normalize(&cross(&dir, &self.up));
        self.eye += right * amount;
        self.target += right * amount;
    }

    /// Déplacement vertical absolu (vers le haut ou le bas)
    pub fn move_up(&mut self, amount: f32) {
        let world_up = vec3(0.0, 1.0, 0.0);
        self.eye += world_up * amount;
        self.target += world_up * amount;
    }

    // NOUVEAU : Calcule l'orientation à partir des mouvements relatifs de la souris
    pub fn rotate_mouse(&mut self, delta_x: f32, delta_y: f32, sensitivity: f32) {
        self.yaw += delta_x * sensitivity;
        self.pitch += delta_y * sensitivity; // Inversé pour que glisser la souris vers le haut lève les yeux

        // Limite le pitch à ~85 degrés pour éviter que la caméra ne se retourne complètement
        let max_pitch: f32 = 1.484;
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

        self.update_target();
    }
}

// ─── Lumière directionnelle ───────────────────────────────────────────────────

pub struct LightController {
    pub position: Vec3,
    pub base_color: Vec3,
    pub intensity: f32,
}

impl LightController {
    pub fn new(position: Vec3, color: Vec3) -> Self {
        Self {
            position,
            base_color: color,
            intensity: 1.0,
        }
    }

    pub fn effective_color(&self) -> [f32; 3] {
        [
            (self.base_color.x * self.intensity).min(1.0),
            (self.base_color.y * self.intensity).min(1.0),
            (self.base_color.z * self.intensity).min(1.0),
        ]
    }

    pub fn position_vec4(&self) -> [f32; 4] {
        [self.position.x, self.position.y, self.position.z, 1.0]
    }

    pub fn move_x(&mut self, amount: f32) {
        self.position.x += amount;
    }

    pub fn move_y(&mut self, amount: f32) {
        self.position.y += amount;
    }

    pub fn move_z(&mut self, amount: f32) {
        self.position.z += amount;
    }

    pub fn change_intensity(&mut self, delta: f32) {
        self.intensity = (self.intensity + delta).clamp(0.0, 3.0);
    }
}