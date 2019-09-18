use crate::consts::*;
use crate::trace::Ray;

use glam::{ vec2, Vec2, Vec3 };

use rand::prelude::*;

pub trait Light: Send + Sync {
    fn get_ray(&self, seed: usize, rng: &mut ThreadRng) -> Ray;
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub pos: Vec2,
    pub spectrum: Vec3,
    pub angle: f32,
}

impl PointLight {
    pub const ANGLE_PER_SAMPLE: f32 = 2.0 * std::f32::consts::PI / RAYS_PER_SAMPLE as f32;
}

impl Light for PointLight {
    fn get_ray(&self, ray_number: usize, rng: &mut ThreadRng) -> Ray {
        let randomness = rng.gen::<f32>() * PointLight::ANGLE_PER_SAMPLE;
        let angle = ray_number as f32 * PointLight::ANGLE_PER_SAMPLE - 0.5 * PointLight::ANGLE_PER_SAMPLE + randomness;
        Ray::new(self.pos, vec2(angle.cos(), angle.sin()), self.spectrum / RAYS_PER_SAMPLE as f32, 1.0)
    }
}