// camera.rs
use nalgebra_glm::{Vec3, vec3, look_at, normalize, cross, rotation};
use nalgebra_glm::TMat4;

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            eye: vec3(0.0, 0.0, 0.5),
            target: vec3(0.0, 0.0, 0.0),
            up: vec3(0.0, 1.0, 0.0),
        }
    }

    pub fn view_matrix(&self) -> TMat4<f32> {
        look_at(&self.eye, &self.target, &self.up)
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

    /// Déplacement vertical absolu (vers le haut ou le bas) - AJOUTÉ
    pub fn move_up(&mut self, amount: f32) {
        let world_up = vec3(0.0, 1.0, 0.0);
        self.eye += world_up * amount;
        self.target += world_up * amount;
    }

    pub fn rotate_yaw(&mut self, angle_radians: f32) {
        let dir = self.target - self.eye;
        let rotated_dir = rotate_y_axis(&dir, angle_radians);
        self.target = self.eye + rotated_dir;
    }

    /// Rotation verticale (pitch) : regarde en haut ou en bas.
    pub fn rotate_pitch(&mut self, angle_radians: f32) {
        let forward = normalize(&(self.target - self.eye));
        let right = normalize(&cross(&forward, &self.up));

        // Angle actuel par rapport à l'horizontale
        let current_pitch = forward.y.asin();

        // Clamp : on ne laisse pas dépasser ±85° (1.484 radians)
        let max_pitch: f32 = 1.484;
        let new_pitch = (current_pitch + angle_radians).clamp(-max_pitch, max_pitch);
        let delta = new_pitch - current_pitch;

        if delta.abs() < 1e-6 {
            return;
        }

        let rot = rotation(delta, &right);
        let dir = self.target - self.eye;
        let rotated = (rot * dir.push(1.0)).xyz();
        self.target = self.eye + rotated;
    }
}

pub fn rotate_y_axis(vector: &Vec3, angle: f32) -> Vec3 {
    let rot_matrix = rotation(angle, &vec3(0.0, 1.0, 0.0));
    let ve4 = vector.push(1.0);
    (rot_matrix * ve4).xyz()
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