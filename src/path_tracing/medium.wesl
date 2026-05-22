import package::path_tracing::general::{ sample_vndf_ggx, from_onb, };
import package::importance_sampling::shared::{ rand_f32, to_onb, cosine_weighted_to_oren_nayer_improved };

fn get_normal(ray_direction:vec3<f32>, normal:vec3<f32>, tangent:vec3<f32>, seed:u32, roughness:vec2<f32>) -> vec3<f32> {
    let ray_onb = to_onb(ray_direction, normal, tangent);
    let normal_onb = sample_vndf_ggx(seed, ray_onb, roughness);
    return from_onb(normal_onb, normal, tangent);
}

fn make_diffuse(hit_color:vec3<f32>, ray:Ray, intersection_normal:vec3<f32>, tangent:vec3<f32>, roughness:f32, new_dir: vec3<f32>) -> vec3<f32> {
    let ray_onb = to_onb(ray.direction, intersection_normal, tangent);
    return cosine_weighted_to_oren_nayer_improved(ray_onb, new_dir, roughness, hit_color);
}

fn conv_rough(rough:f32) -> f32 {
    return rough;
}

fn pick_ref_idx_blend(picked: ptr<function, PickedRefIdx>, seed:u32) -> vec3<f32> {
    let picked_idx = rand_f32(seed);
    *picked = PickedRefIdx(picked_idx);
    return vec3(1.0);
}

fn pick_ref_idx(picked: ptr<function, PickedRefIdx>, low_idx:f32, high_idx:f32, seed:u32) -> f32 {
    return mix(low_idx, high_idx, (*picked).picked_idx);
}