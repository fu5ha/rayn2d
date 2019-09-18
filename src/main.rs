use glam::{ vec2, vec3, Vec3 };

use minifb::{ Window, WindowOptions, Key };

use rayon::prelude::*;

use rand::prelude::*;

mod consts;
mod draw;
mod light;
mod material;
mod object;
mod trace;
mod world;
mod sdf;

use consts::*;
use draw::*;
use light::*;
use material::*;
use object::*;
use trace::*;
use world::*;

fn setup_world() -> World {

    let red_diffuse = Lambertian {
        ior: 0.5,
        color: vec3(0.7, 0.2, 0.3),
    };
    let blue_diffuse = Lambertian {
        ior: 0.95,
        color: vec3(0.5, 0.2, 0.9),
    };

    let objects: Vec<Box<dyn WorldObject>> = vec![
        Box::new(Circle {
            center: vec2(WIDTH as f32 / 2.0, HEIGHT as f32 / 4.0),
            radius: HEIGHT as f32 / 8.0,
            material: Box::new(red_diffuse.clone()),
            uuid: 0
        }),
        Box::new(Circle {
            center: vec2(WIDTH as f32 / 2.0, HEIGHT as f32 - HEIGHT as f32 / 4.0),
            radius: HEIGHT as f32 / 8.0,
            material: Box::new(blue_diffuse.clone()),
            uuid: 0
        }),
    ];

    let lights: Vec<Box<dyn Light>> = vec![
        Box::new(PointLight {
            pos: vec2(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0),
            spectrum: vec3(1.0, 0.4, 0.2),
            angle: 0.0,
        }),
        Box::new(PointLight {
            pos: vec2(WIDTH as f32 / 8.0, HEIGHT as f32 / 8.0),
            spectrum: vec3(0.5, 0.4, 1.2),
            angle: 0.0,
        }),
        Box::new(PointLight {
            pos: vec2(WIDTH as f32 - WIDTH as f32 / 8.0, HEIGHT as f32 / 8.0),
            spectrum: vec3(1.0, 1.5, 0.6),
            angle: 0.0,
        }),
    ];

    World { objects, lights }
}

fn main() {
    let mut display_buf: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut img_buf: Vec<Vec3> = vec![vec3(0.0, 0.0, 0.0); WIDTH * HEIGHT];

    let mut window = Window::new(
        "rayn2d",
        WIDTH,
        HEIGHT,
        WindowOptions::default()).unwrap();

    let mut world = setup_world();
    
    let mut draw_instructions: Vec<DrawInstruction> = Vec::new();
    
    while window.is_open() && !window.is_key_down(Key::Escape) {
        update(&mut world, &mut draw_instructions);

        draw(&draw_instructions, &mut img_buf);
        draw_instructions.clear();

        update_display(&mut img_buf, &mut display_buf);

        window.update_with_buffer(&display_buf).unwrap();
    }
}

fn update(world: &mut World, draw_instructions: &mut Vec<DrawInstruction>) {
    for light in world.lights.iter() {
        draw_instructions.par_extend(
            rayon::iter::repeat(light).zip(0..RAYS_PER_UPDATE).flat_map(|(light, _ray_number)| {
                let seed = rand::thread_rng().gen::<f32>();

                trace(world, light.get_ray(seed)).into_par_iter()
            })
        );
        // draw_instructions.extend(
        //     std::iter::repeat(light).zip(0..RAYS_PER_UPDATE).flat_map(|(light, ray_number)| {
        //         let angle = light.angle + ray_number as f32 * EPS_ANGLE;
        //         if angle > std::f32::consts::PI * 2.0 {
        //             return Vec::new().into_iter()
        //         }
        //         let ray = Ray::new(light.pos, vec2(angle.cos(), angle.sin()).normalize(), light.spectrum, 1.0);

        //         trace::trace(world, ray).into_iter()
        //     })
        // );
    }
}


