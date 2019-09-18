use glam::{ vec2, vec3, Vec3 };

use minifb::{ Window, WindowOptions, Key };

use rayon::prelude::*;

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
            spectrum: vec3(1.0, 0.4, 0.2) * 50.0,
            angle: 0.0,
        }),
        Box::new(PointLight {
            pos: vec2(WIDTH as f32 / 8.0, HEIGHT as f32 / 8.0),
            spectrum: vec3(0.5, 0.4, 1.2) * 150.0,
            angle: 0.0,
        }),
        Box::new(PointLight {
            pos: vec2(WIDTH as f32 - WIDTH as f32 / 8.0, HEIGHT as f32 / 8.0),
            spectrum: vec3(1.0, 1.5, 0.6) * 150.0,
            angle: 0.0,
        }),
    ];

    World { objects, lights }
}

fn main() {
    let mut display_buf: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut final_img_buf: Vec<Vec3> = vec![vec3(0.0, 0.0, 0.0); WIDTH * HEIGHT];
    let mut scratch_img_buf: Vec<Vec3> = vec![vec3(0.0, 0.0, 0.0); WIDTH * HEIGHT];

    let mut window = Window::new(
        "rayn2d",
        WIDTH,
        HEIGHT,
        WindowOptions::default()).unwrap();

    let mut world = setup_world();
    
    let mut draw_instructions: Vec<DrawInstruction> = Vec::new();

    let mut rng = rand::thread_rng();
    let mut tracer_state = TracerState {
        current_ray_count: 0,
        ray_index_vec: rand::seq::index::sample(&mut rng, RAYS_PER_SAMPLE, RAYS_PER_SAMPLE).into_vec(),
        current_sample_count: 0,
    };
    
    while window.is_open() && !window.is_key_down(Key::Escape) {
        update(&mut tracer_state, &mut world, &mut draw_instructions);

        draw(&draw_instructions, &mut scratch_img_buf);
        draw_instructions.clear();

        update_display(&tracer_state, &mut final_img_buf, &mut scratch_img_buf, &mut display_buf);
        window.update_with_buffer(&display_buf).unwrap();

        if tracer_state.current_ray_count == RAYS_PER_SAMPLE {
            consolidate(&mut final_img_buf, &mut scratch_img_buf, tracer_state.current_sample_count);
            tracer_state.current_ray_count = 0;
            tracer_state.ray_index_vec = rand::seq::index::sample(&mut rng, RAYS_PER_SAMPLE, RAYS_PER_SAMPLE).into_vec();
            tracer_state.current_sample_count += 1;
        }
    }
}

fn update(state: &mut TracerState, world: &mut World, draw_instructions: &mut Vec<DrawInstruction>) {
    let indices = state.ray_index_vec.split_off(state.ray_index_vec.len() - RAYS_PER_UPDATE);
    for light in world.lights.iter() {
        // Multi-threaded
        draw_instructions.par_extend(
            rayon::iter::repeat(light).zip(indices.par_iter()).flat_map(|(light, ray_number)| {
                let mut rng = rand::thread_rng();
                trace(world, light.get_ray(*ray_number, &mut rng)).into_par_iter()
            })
        );

        // Single-threaded
        // draw_instructions.extend(
        //     std::iter::repeat(light).zip(indices.iter()).flat_map(|(light, ray_number)| {
        //         let mut rng = rand::thread_rng();
        //         trace(world, light.get_ray(*ray_number, &mut rng)).into_iter()
        //     })
        // );
    }
    state.current_ray_count += RAYS_PER_UPDATE;
}


