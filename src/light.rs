use crate::trace::Ray;

use glam::{ vec2, Vec2, Vec3 };

pub trait Light: Send + Sync {
    fn get_ray(&self, seed: f32) -> Ray;
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub pos: Vec2,
    pub spectrum: Vec3,
    pub angle: f32,
}

impl Light for PointLight {
    fn get_ray(&self, seed: f32) -> Ray {
        let angle = seed * 2.0 * std::f32::consts::PI;
        Ray::new(self.pos, vec2(angle.cos(), angle.sin()), self.spectrum, 1.0)
    }
}