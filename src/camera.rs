use nalgebra_glm::{Vec3, vec3, look_at, normalize, cross, rotation};
use nalgebra_glm::TMat4;

pub struct Camera {
    pub eye : Vec3,
    pub target : Vec3,
    pub up : Vec3,
}

impl Camera {
    pub fn new() -> Self{
        Self{
            eye : vec3(0.0, 0.0, 0.5),
            target : vec3(0.0, 0.0, 0.0),
            up : vec3(0.0, 1.0, 0.0),
        } 
    }

    pub fn view_matrix(&self) -> TMat4<f32>{
        look_at(&self.eye, &self.target, &self.up)
    }

    // pub fn move_foward(&mut self, amount : f32){
    //     let forward = normalize(&(self.target - self.eye));
    //     let right = normalize(&cross(&forward, &self.up));
    //     self.eye += right * amount;
    //     //self.target += right * amount;
    // }    

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

    pub fn rotate_yaw(&mut self, angle_radians: f32) {
        let dir = self.target - self.eye;
        let rotated_dir = rotate_y_axis(&dir, angle_radians);
        self.target = self.eye + rotated_dir;
    }


}


pub fn rotate_y_axis(vector: &Vec3, angle: f32) -> Vec3 {
    let rot_matrix = rotation(angle, &vec3(0.0, 1.0, 0.0)); // Y axis
    let ve4 = vector.push(1.0);

    (rot_matrix * ve4).xyz()
}