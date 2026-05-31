use winit::event::{ElementState, VirtualKeyCode};

#[derive(Default)]
pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub strafe_left: bool,
    pub strafe_right: bool,
    pub move_up: bool,
    pub move_down: bool,
    pub light_left: bool,
    pub light_right: bool,
    pub light_up: bool,
    pub light_down: bool,
    pub light_forward: bool,
    pub light_backward: bool,
    pub light_inc_intensity: bool,
    pub light_dec_intensity: bool,
}

impl InputState {
    pub fn update(&mut self, keycode: VirtualKeyCode, state: ElementState) {
        let is_pressed = state == ElementState::Pressed;
        match keycode {
            VirtualKeyCode::Up => self.forward = is_pressed,
            VirtualKeyCode::Down => self.backward = is_pressed,
            VirtualKeyCode::Right => self.strafe_right = is_pressed,
            VirtualKeyCode::Left => self.strafe_left = is_pressed,
            VirtualKeyCode::Space => self.move_down = is_pressed,
            VirtualKeyCode::LShift | VirtualKeyCode::RShift => self.move_up = is_pressed,
            VirtualKeyCode::O => self.light_left = is_pressed,
            VirtualKeyCode::L => self.light_right = is_pressed,
            VirtualKeyCode::K => self.light_up = is_pressed,
            VirtualKeyCode::M => self.light_down = is_pressed,
            VirtualKeyCode::N => self.light_forward = is_pressed,
            VirtualKeyCode::J => self.light_backward = is_pressed,
            VirtualKeyCode::I => self.light_inc_intensity = is_pressed,
            VirtualKeyCode::P => self.light_dec_intensity = is_pressed,
            _ => {}
        }
    }
}