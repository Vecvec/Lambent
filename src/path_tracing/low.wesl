import package::path_tracing::general::{sample_vndf_ggx, from_onb, PickedNormal,};
import package::importance_sampling::shared::{ rand_f32, to_onb, cosine_weighted_to_oren_nayer_improved };

fn get_normal(ray_direction: vec3<f32>, normal: vec3<f32>, tangent: vec3<f32>, seed: u32, roughness: vec2<f32>) -> PickedNormal {
    return PickedNormal(normal, 1.0, vec3<f32>(0.0, 1.0, 0.0), vec2(0.0));
}

fn get_normal_bsdf(in: vec3<f32>, out: vec3<f32>, picked_normal: PickedNormal, normal: vec3<f32>, tangent: vec3<f32>) -> f32 {
    return 1.0;
}

fn make_diffuse(hit_color: vec3<f32>, ray: Ray, intersection_normal: vec3<f32>, tangent: vec3<f32>, roughness: f32, new_dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0);
}

fn conv_rough(rough: f32) -> f32 {
    return 0.0;
}

fn pick_ref_idx_blend(picked: ptr<function, PickedRefIdx>, seed: u32) -> vec3<f32> {
    // no-op
    return vec3(1.0);
}

fn pick_ref_idx(picked: ptr<function, PickedRefIdx>, low_idx: f32, high_idx: f32, seed: u32) -> f32 {
    return low_idx;
}