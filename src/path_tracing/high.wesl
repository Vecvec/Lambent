import package::path_tracing::general::{ sample_vndf_ggx, from_onb, };
import package::importance_sampling::shared::{ rand_f32, to_onb, cosine_weighted_to_oren_nayer_improved };

fn get_normal(ray_direction:vec3<f32>, normal:vec3<f32>, tangent:vec3<f32>, seed:u32, roughness:vec2<f32>) -> vec3<f32> {
    let ray_onb = to_onb(ray_direction, normal, tangent);
    let normal_onb = sample_vndf_ggx(seed, ray_onb, roughness);
    return from_onb(normal_onb, normal, tangent);
}

fn make_diffuse(hit_color:vec3<f32>, ray:Ray, intersection_normal:vec3<f32>, tangent:vec3<f32>, roughness:f32, new_dir: vec3<f32>) -> vec3<f32> {
    return cosine_weighted_to_oren_nayer_improved(ray.direction, intersection_normal, new_dir, roughness, hit_color);
}

fn conv_rough(rough:f32) -> f32 {
    return rough;
}

const LOW_LAMBDA: f32 = 760.0;
const HIGH_LAMBDA: f32 = 340.0;

fn pick_ref_idx_blend(picked: ptr<function, PickedRefIdx>, seed:u32) -> vec3<f32> {
    let picked_idx = rand_f32(seed);
    *picked = PickedRefIdx(picked_idx);
    let lambda = mix(LOW_LAMBDA, HIGH_LAMBDA, picked_idx);
    let x = fma(1.056, g(lambda, 599.8, 0.0264, 0.0323), fma(0.362, g(lambda, 442.0, 0.0624, 0.0374), -(0.065 * g(lambda, 501.1, 0.049, 0.0382))));
    let y = fma(0.821, g(lambda, 568.8, 0.0213, 0.0248), (0.286 * g(lambda, 530.9, 0.0613, 0.0322)));
    let z = fma(1.217, g(lambda, 437.0, 0.0845, 0.0278), (0.681 * g(lambda, 459.0, 0.0385, 0.0725)));
    let rgb = to_linear(vec3<f32>(x, y, z));
    // These seem random, but are the reprocals of the intergrals of this function so it matches all other shaders
    return saturate(rgb) * vec3<f32>(3.910, 4.411, 5.511);
}

fn pick_ref_idx(picked: ptr<function, PickedRefIdx>, low_idx:f32, high_idx:f32, seed:u32) -> f32 {
    return mix(low_idx, high_idx, picked.picked_idx);
}

fn g(x:f32, u:f32, tau_1:f32, tau_2:f32) -> f32 {
    var tau: f32;
    if (x < u) {
        tau = tau_1;
    } else {
        tau = tau_2;
    }
    let x_minus_u = x - u;
    let x_minus_u_sqrd = x_minus_u * x_minus_u;
    let tau_sqrd = tau * tau;
    return exp(-(x_minus_u_sqrd * tau_sqrd) * 0.5);
}

//https://en.wikipedia.org/wiki/SRGB transposed (since wgsl has its matricies the other way round)
const CIE_TO_RGB = mat3x3<f32>(
    vec3<f32>(3.2406, -0.9689, 0.0557),
    vec3<f32>(-1.5372, 1.8758, -0.204),
    vec3<f32>(-0.4986, 0.0415, 1.057),
);

fn to_linear(cie:vec3<f32>) -> vec3<f32> {
    return CIE_TO_RGB * cie;
}