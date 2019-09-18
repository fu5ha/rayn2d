use crate::consts::*;
use crate::trace::Ray;

use glam::{ vec2, Vec2, Vec3 };

pub trait Light: Send + Sync {
    fn get_ray(&self, seed: usize, randomness: f32) -> Ray;
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub pos: Vec2,
    pub spectrum: Vec3,
    pub angle: f32,
}

impl PointLight {
    pub const ANGLE_PER_RAY: f32 = 2.0 * std::f32::consts::PI / RAYS_PER_SAMPLE as f32;
}

impl Light for PointLight {
    fn get_ray(&self, ray_number: usize, randomness: f32) -> Ray {
        let randomness = randomness * PointLight::ANGLE_PER_RAY;
        let angle = ray_number as f32 * PointLight::ANGLE_PER_RAY - 0.5 * PointLight::ANGLE_PER_RAY + randomness;
        Ray::new(self.pos, vec2(angle.cos(), angle.sin()), self.spectrum / RAYS_PER_SAMPLE as f32 * (WIDTH as f32).sqrt(), 1.0)
    }
}