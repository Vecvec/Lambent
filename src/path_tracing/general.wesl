import package::default_mode::{ get_normal, conv_rough, make_diffuse, pick_ref_idx, pick_ref_idx_blend, get_normal_bsdf };
import package::importance_sampling::shared::{ Sample, to_onb, copy_sign, rand_xz_vec3_f32, safe_div_vec3, vec3_near_zero, p_hat, update, safe_div, sam_from_res, from_info, u32_to_normalised, get_light_dir, update_marcov_state, grid_loc, grid_load, grid_store_markov_chain, strip_NaN_vec3, normalised_to_u32, TAU, DIFFUSE, METALLIC, TRANSPARENT, SKY, U32_MAX, CACHE_TIMEOUT, };
import package::bindings::{
    // Types
    MarcovChainState,
    Reservoir,
    // aliases
    ray_query_maybe_vertex_return,
    // bindings
    world_markov_chains,
    old_world_markov_chains,
    markov_chains,
    old_markov_chains,
    gi_reservoirs,
    old_gi_reservoirs,
    output,
    output_normal,
    output_albedo,
    camera,
    acc_struct,
    materials,
    material_idx,
    textures,
    tex_recolour,
    tex_pos,
    bg,
    sam,
    info,
    // functions
    unpack_2xu16,
    near_zero,
    // constants
    INVALID_KEY,
    STATE_LOCKED,
    STATE_UNLOCKED,
    MAX_NUM_SAMPLES,
    // Overrides
    SAMPLES,
};
import package::path_tracing::weighted_reservoir_sampling;

@if(!vertex_return)
import package::bindings::{ get_vertices, };

enable wgpu_ray_query;
enable wgpu_binding_array;
@if(vertex_return)
enable wgpu_ray_vertex_return;

// WIP: Importance sampling

/*fn jacobian_determinant(current:Sample, other:Sample) -> f32 {
    let q1_to_q2 = other.sample_point - other.visible_point;
    let r1_to_q2 = current.sample_point - other.visible_point;
    let cos_theta_r = dot(other.visible_normal, r1_to_q2);
    let cos_theta_q = dot(other.visible_normal, q1_to_q2);
    return (clamp(abs(cos_theta_r), 0.01, 1.0) / clamp(abs(cos_theta_q), 0.01, 1.0)) * (len_sqrd(q1_to_q2) / len_sqrd(r1_to_q2));
}*/

// adds a light to the current frames list of lights
/*fn push(light: vec4<f32>) {
    if (near_zero(light.w)) {
        return;
    }
    //let old_len = atomicLoad(&lights.size);
    let val = atomicAdd(&lights.size, 1u);
    if val >= arrayLength(&lights.lights) - 1u {
        atomicStore(&lights.size, arrayLength(&lights.lights) - 1u);
    } else {
        lights.lights[val] = light;
    }
    return;
}

// gets a random light
fn get_rand(seed:u32) -> vec4<f32> {
    let maximum_new = atomicLoad(&lights.size);
    let maximum_old = atomicLoad(&old_lights.size);
    let maximum = maximum_old + maximum_new;
    let val = rand_u32(seed) % maximum;
    if val < maximum_new {
        let light = lights.lights[val];
        let brightness = brightness(f32(maximum), light.w);
        return vec4<f32>(light.xyz, brightness);
    } else {
        let light = old_lights.lights[val];
        let brightness = brightness(f32(maximum), light.w);
        return vec4<f32>(light.xyz, brightness);
    }
}*/

fn grid_load_radiance_estimate(pos: vec3<f32>, normal: vec3<f32>, incoming: vec3<f32>) -> vec3<f32> {
    let grid_pos = grid_loc(pos, camera_loc);
    let hash = primary_hasher(grid_pos, normal, incoming);
    let idx = hash % arrayLength(&world_markov_chains);
    var num_samples = atomicLoad(&world_markov_chains[idx].num_samples);
    var float_radiance = vec3<f32>(bitcast<f32>(atomicLoad(&world_markov_chains[idx].luminance[0])), bitcast<f32>(atomicLoad(&world_markov_chains[idx].luminance[1])), bitcast<f32>(atomicLoad(&world_markov_chains[idx].luminance[2])));
    let secondary_hash = atomicLoad(&world_markov_chains[idx].secondary_hash);
    if (secondary_hash != secondary_hasher(grid_pos, normal, incoming)) {
        if (atomicLoad(&old_world_markov_chains[idx].secondary_hash) == secondary_hasher(grid_pos, normal, incoming)) {
            num_samples = atomicLoad(&old_world_markov_chains[idx].num_samples);
            float_radiance = vec3<f32>(bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].luminance[0])), bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].luminance[1])), bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].luminance[2])));
        } else {
            return vec3(0.0);
        }
    } else {
        world_markov_chains[idx].timeout = CACHE_TIMEOUT;
    }
    if (num_samples == 0) {
        return vec3(0.0);
    }
    let radiance = float_radiance / f32(num_samples);
    let clamped_radiance = clamp(radiance, vec3(0.0), vec3(1000.0));
    return clamped_radiance;
}

fn grid_add_radiance(pos:vec3<f32>, normal:vec3<f32>, incoming: vec3<f32>, radiance: vec3<f32>) {
    let grid_pos = grid_loc(pos, camera_loc);
    let hash = primary_hasher(grid_pos, normal, incoming);
    let idx = hash % arrayLength(&world_markov_chains);
    let secondary_hash = secondary_hasher(grid_pos, normal, incoming);
    let clamped_radiance = clamp(radiance, vec3(0.0), vec3(100.0));
    if (atomicLoad(&world_markov_chains[idx].secondary_hash) != secondary_hash) {
        if !(atomicCompareExchangeWeak(&world_markov_chains[idx].secondary_hash, INVALID_KEY, secondary_hash).exchanged) {
            return;
        }
    }
    let locked = atomicExchange(&world_markov_chains[idx].lock, STATE_LOCKED);
    // We must make forward progress even if this is locked, so we just return.
    if locked == STATE_LOCKED {
        return;
    }
    /*if (atomicLoad(&world_markov_chains[idx].secondary_hash) == INVALID_KEY) {
        atomicStore(&world_markov_chains[idx].secondary_hash, secondary_hash);
    }
    if (atomicLoad(&world_markov_chains[idx].secondary_hash) != secondary_hash) {
        atomicStore(&world_markov_chains[idx].lock, STATE_UNLOCKED);
        return;
    }*/
    let num_samples = atomicLoad(&world_markov_chains[idx].num_samples);
    if (num_samples >= MAX_NUM_SAMPLES) {
        atomicStore(&world_markov_chains[idx].lock, STATE_UNLOCKED);
        world_markov_chains[idx].timeout = CACHE_TIMEOUT;
        return;
    }
    atomicAdd(&world_markov_chains[idx].num_samples, 1);
    let radiance_red = strip_NaN(clamped_radiance.x + bitcast<f32>(atomicLoad(&world_markov_chains[idx].luminance[0])));
    atomicStore(&world_markov_chains[idx].luminance[0], bitcast<u32>(max(radiance_red, 0.0)));
    let radiance_green = strip_NaN(clamped_radiance.y + bitcast<f32>(atomicLoad(&world_markov_chains[idx].luminance[1])));
    atomicStore(&world_markov_chains[idx].luminance[1], bitcast<u32>(max(radiance_green, 0.0)));
    let radiance_blue = strip_NaN(clamped_radiance.z + bitcast<f32>(atomicLoad(&world_markov_chains[idx].luminance[2])));
    atomicStore(&world_markov_chains[idx].luminance[2], bitcast<u32>(max(radiance_blue, 0.0)));
    atomicStore(&world_markov_chains[idx].lock, STATE_UNLOCKED);
    world_markov_chains[idx].timeout = CACHE_TIMEOUT;
}

fn brightness(max:f32, odds: f32) -> f32 {
    return odds;
    //return odds * 32.0;
    //return odds * 400.0;
    //return odds * 4.0;
    //return odds * 80.0;
    //return 4.0;
}

var<immediate> seed_offset:u32;

override IS_SAMPLES = 64u;

const RESAMPLE_MIN = 50u;
const RESAMPLE_MAX = 100u;

// TODO: Move this into a inter pixel buffer so all pixels can sample from it.
// S in the paper.
var<private> tentative_marcov_chain_state: MarcovChainState;

// Sc in the paper.
var<private> current_marcov_chain_state: MarcovChainState;

struct MarcovChainHolder {
    state: array<MarcovChainState, NUM_RESAMPLES>,
    sum_weight_sum: f32,
}

const NUM_RESAMPLES_SCREEN_SPACE = 1;
const NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME = 2;
const NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME_CLOSE = 1;
const NUM_RESAMPLES_WORLD_SPACE = 4;
// const NUM_RESAMPLES_SCREEN_SPACE = 0;
// const NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME = 0;
// const NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME_CLOSE = 0;
// const NUM_RESAMPLES_WORLD_SPACE = 4;
const NUM_NON_PIXEL_RESAMPLES = NUM_RESAMPLES_SCREEN_SPACE + NUM_RESAMPLES_WORLD_SPACE + NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME + NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME_CLOSE;
const NUM_RESAMPLES = NUM_NON_PIXEL_RESAMPLES + 1;
//const NUM_NON_PIXEL_RESAMPLES = 0;
var<private> other_pixels_marcov_chain_states: MarcovChainHolder;

fn shuffle_marcov_chain(next_subgroup:u32) {
    tentative_marcov_chain_state = get_tentative_marcov_chain(next_subgroup);
}

fn get_tentative_marcov_chain(subgroup_id:u32) -> MarcovChainState {
    return MarcovChainState(
        subgroupShuffle(tentative_marcov_chain_state.light_source, subgroup_id),
        subgroupShuffle(tentative_marcov_chain_state.mean_cosine, subgroup_id),
        subgroupShuffle(tentative_marcov_chain_state.weight_sum, subgroup_id),
        subgroupShuffle(tentative_marcov_chain_state.num_samples, subgroup_id),
        subgroupShuffle(tentative_marcov_chain_state.score, subgroup_id),
    );
}

const RESAMPLE_SIZE: u32 = 91;
const RESAMPLE_SIZE_CLOSE: u32 = 5;

const RESAMPLE_SIZE_SHIFT: i32 = i32(RESAMPLE_SIZE) / 2;
const RESAMPLE_SIZE_SHIFT_CLOSE: i32 = i32(RESAMPLE_SIZE_CLOSE) / 2;

fn fill_mc_screen_space_states(own_seed: ptr<function, u32>, sub_size: u32, pix_pos: vec2<u32>, size: vec2<u32>) {
    var i = 0;
    for (; i < NUM_RESAMPLES_SCREEN_SPACE; i++) {
        let sub_id = rand_u32(*own_seed) % sub_size;
        other_pixels_marcov_chain_states.state[i] = get_tentative_marcov_chain(sub_id);
    }
    for (; i < (NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME + NUM_RESAMPLES_SCREEN_SPACE); i++) {
        let x_rand = rand_u32(*own_seed);
        *own_seed = x_rand;
        let y_rand = rand_u32(*own_seed);
        *own_seed = y_rand;
        let x = i32(x_rand % RESAMPLE_SIZE) - RESAMPLE_SIZE_SHIFT;
        let y = i32(y_rand % RESAMPLE_SIZE) - RESAMPLE_SIZE_SHIFT;
        // TODO: what about out of range pixels? Should we skip them?
        let neighbour = vec2<u32>(vec2<i32>(pix_pos) + vec2<i32>(x, y));
        let idx = neighbour.x + (neighbour.y * size.x);
        other_pixels_marcov_chain_states.state[i] = old_markov_chains[idx];
    }
    for (; i < (NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME + NUM_RESAMPLES_SCREEN_SPACE + NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME_CLOSE); i++) {
        let x_rand = rand_u32(*own_seed);
        *own_seed = x_rand;
        let y_rand = rand_u32(*own_seed);
        *own_seed = y_rand;
        let x = i32(x_rand % RESAMPLE_SIZE_CLOSE) - RESAMPLE_SIZE_SHIFT_CLOSE;
        let y = i32(y_rand % RESAMPLE_SIZE_CLOSE) - RESAMPLE_SIZE_SHIFT_CLOSE;
        // TODO: what about out of range pixels? Should we skip them?
        let neighbour = vec2<u32>(vec2<i32>(pix_pos) + vec2<i32>(x, y));
        let idx = neighbour.x + (neighbour.y * size.x);
        other_pixels_marcov_chain_states.state[i] = old_markov_chains[idx];
    }
}

const JITTER_SIZE = GRID_SIZE * 1.5;
//const JITTER_SIZE = 0.0;
fn fill_mc_world_space_states(own_seed: ptr<function, u32>, pos: vec3<f32>, normal: vec3<f32>, incoming: vec3<f32>) {
    for (var i = (NUM_RESAMPLES_SCREEN_SPACE + NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME + NUM_RESAMPLES_SCREEN_SPACE_LAST_FRAME_CLOSE); i < NUM_NON_PIXEL_RESAMPLES; i++) {
        let force_self = i == 0;
        let rand_candidate = rand_f32(*own_seed) < 0.7 && !force_self;
        let jitter_mul = select(select(1.0, 0.75, rand_candidate), 0.0, force_self);
        let jitter = rand_xz_vec3_f32(own_seed)*JITTER_SIZE*jitter_mul;
        let s = grid_load(pos + from_onb_no_tangent(jitter, normal), normal, incoming, camera_loc, rand_candidate);
        other_pixels_marcov_chain_states.state[i] = s;
        *own_seed++;
    }
    other_pixels_marcov_chain_states.state[NUM_NON_PIXEL_RESAMPLES] = tentative_marcov_chain_state;
}

var<private> sum_mc: f32;

fn resample_mcmm(own_seed: ptr<function, u32>, origin: vec3<f32>, normal: vec3<f32>) -> MarcovChainState {
    var s = MarcovChainState();
    sum_mc = 0.0;
    for (var i = 0; i < NUM_RESAMPLES; i++) {
        var rand_candidate = rand_f32(*own_seed);
        *own_seed = rand_u32(*own_seed);
        let s_candidate = other_pixels_marcov_chain_states.state[i];
        sum_mc += s_candidate.weight_sum;

        let mean = get_light_dir(s, origin);
        let weight = s_candidate.weight_sum;// * select(0.0, 1.0, dot(mean, normal) > 0);

        var rand = rand_f32(*own_seed);
        if (rand * sum_mc) < weight {
            s = s_candidate;
        }
        *own_seed = rand_u32(*own_seed);
    }
    other_pixels_marcov_chain_states.sum_weight_sum = sum_mc;
    //s = MarcovChainState();
    return s;
}

fn marcov_chain_valid(state: MarcovChainState) -> bool {
    return state.weight_sum > 0.0;
}

const FLOAT_MAX = 3.40282347E+38;

// x0 is never used so we don't include it.
//fn update_marcov_state(/* x0: vec3<f32>, */ x1: vec3<f32>, x2: vec3<f32>, pdf: f32, p_hat: f32, seed: u32, not_vmf: bool) {
//    let score = safe_div(p_hat, pdf);
//    let a = min(1.0, div_or(score, current_marcov_chain_state.score, 1.0));
//    let rand = rand_f32(seed);
//    var update = false;
//    if ((score > 0.0) && (!marcov_chain_valid(current_marcov_chain_state) || rand < a)) { // May want to convert this to be closer to the GI update.
//        if not_vmf {
//            tentative_marcov_chain_state = MarcovChainState();
//        }
//        tentative_marcov_chain_state.score = score;
//        update = true;
//    }
//    tentative_marcov_chain_state.num_samples = min(tentative_marcov_chain_state.num_samples + 1, MARCOV_CHAIN_SAMPLE_CLAMP);
//    let alpha = max(1.0 / f32(tentative_marcov_chain_state.num_samples), 0.1);
//    // WGSL mix is the same as the one in the paper.
//    let weight_sum = mix(tentative_marcov_chain_state.weight_sum, score, alpha);
//    tentative_marcov_chain_state.light_source = mix(tentative_marcov_chain_state.weight_sum * tentative_marcov_chain_state.light_source, score * x2, alpha) / weight_sum;
//    let mean = normalize(tentative_marcov_chain_state.light_source - x1);
//    // Instead of mean_cosine there was something else in the paper, but there is no such field in
//    // MarcovChainState and it isn't accessed anywhere else and r isn't accessed here so assuming it's r.
//    let omega = mix(tentative_marcov_chain_state.weight_sum * tentative_marcov_chain_state.mean_cosine * mean, score * normalize(x2 - x1), alpha);
//    tentative_marcov_chain_state.mean_cosine = (length(omega));
//    tentative_marcov_chain_state.weight_sum = weight_sum;
//    if update {
//        current_marcov_chain_state = tentative_marcov_chain_state;
//    }
//}

var<private> camera_loc: vec3<f32>;

const RESTIR_OPTION_NONE: u32 = 0;
const RESTIR_OPTION_PER_SAMPLE: u32 = 1;
const RESTIR_OPTION_PER_PIXEL: u32 = 2;

@workgroup_size(64, 1, 1)
@compute
fn rt_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) work_id: vec3<u32>, @builtin(subgroup_size) sub_size:u32, @builtin(subgroup_invocation_id) sub_id: u32) {
    let screen_size = textureDimensions(output);
    if (id.x > screen_size.x || id.y > screen_size.y) {
        return;
    }
    let idx = id.x + (id.y * screen_size.x);
    tentative_marcov_chain_state = old_markov_chains[idx];
    let x = f32(id.y)/f32(screen_size.y);
    var own_seed = rand_u32((id.x * id.y)) + seed_offset + id.y;
    var pixel_color = vec3<f32>();
    var pixel_normal: vec3<f32>;
    var pixel_albedo: vec3<f32>;
    var pixel_emission: vec3<f32>;
    var pixel_ty: u32;
    var pixel_point: vec3<f32>;
    var pixel_roughness: f32;
    var is_samples = 0u;
    var out_radiance = vec3<f32>();
    var sample: Sample;
    var sample_valid = false;
    var pdf = 0.0;
    let next_sub_id = (sub_id + 1) % sub_size;

    var R = Reservoir();
    for (var i = 0u; i < SAMPLES; i++) {
        fill_mc_screen_space_states(&own_seed, sub_size, id.xy, screen_size);
        let ray_sample = rt_sample(id.xy, own_seed, (i == 0u));
        pixel_normal = ray_sample.normal;
        pixel_albedo = ray_sample.albedo;
        pixel_emission = ray_sample.emission;
        pixel_ty = ray_sample.ty;
        pixel_color = pixel_color + ray_sample.color;
        out_radiance += ray_sample.out_radiance;
        pixel_roughness = ray_sample.roughness;
        pixel_point = ray_sample.point;
        //pixel_color = pixel_color + ray_sample.out_radiance;
        own_seed = rand_u32(own_seed);

        if (ray_sample.color.x < 0.0 || ray_sample.color.y < 0.0 || ray_sample.color.z < 0.0) {
            //textureStore(output, id.xy, vec4<f32>(0.0, 1.0, 0.0, 1.0));
            //return;
        }
        if RESTIR == RESTIR_OPTION_PER_PIXEL {
            if (ray_sample.valid) {
                sample_valid = true;
                sample = Sample(
                    ray_sample.visible_point,
                    normalised_to_u32(ray_sample.visible_normal),
                    ray_sample.point, normalised_to_u32(pixel_normal),
                    out_radiance / f32(SAMPLES),
                    pixel_ty,
                    conv_rough(ray_sample.roughness),
                );
                pdf += ray_sample.pdf;
            }
        } else if RESTIR == RESTIR_OPTION_PER_SAMPLE {
            let sample = Sample(
                ray_sample.visible_point,
                normalised_to_u32(ray_sample.visible_normal),
                ray_sample.point, normalised_to_u32(ray_sample.normal),
                ray_sample.out_radiance,
                ray_sample.ty,
                conv_rough(ray_sample.roughness),
            );
            if (sample.ty != SKY && ray_sample.valid) {
                let w = p_hat(sample); // Don't think `/ pdf` should be here, in the paper it's here but it's probably part of the out radiance.
                R = update(sample, w, R, own_seed, false);
                own_seed = rand_u32(own_seed);
                let confidance = unpack_confidance(R.packed_confidance_valid).confidance;
                R.W = safe_div(R.w, (f32(confidance) * p_hat(sam_from_res(R))));
            } else if sample.ty == SKY {
                R.ty = SKY;
            }
        }
        shuffle_marcov_chain(next_sub_id);
        //lights.samples[idx] = update(sample, 1.0, lights.samples[idx], own_seed);
    }
    pdf /= f32(SAMPLES);
    if RESTIR != RESTIR_OPTION_NONE {
        info[idx] = from_info(Info(camera_loc, pixel_emission, pixel_albedo));
    }
    markov_chains[idx] = tentative_marcov_chain_state;
    textureStore(output_normal, id.xy, vec4<f32>(pixel_normal, 1.0));
    textureStore(output_albedo, id.xy, vec4<f32>(pixel_albedo, 1.0));
    textureStore(output, id.xy, vec4<f32>(max(pixel_color / f32(SAMPLES), vec3<f32>(0.0)), 1.0));

    // put collective samples into GI
    if RESTIR == RESTIR_OPTION_PER_PIXEL {
        if (sample.ty != SKY && sample_valid) {
            let w = p_hat(sample); // Don't think `/ pdf` should be here, in the paper it's here but it's probably part of the out radiance.
            R = update(sample, w, R, own_seed, false);
            own_seed = rand_u32(own_seed);
            let confidance = unpack_confidance(R.packed_confidance_valid).confidance;
            R.W = safe_div(R.w, (f32(confidance) * p_hat(sam_from_res(R))));
        } else if sample.ty == SKY {
            R.ty = SKY;
        }
    }
    gi_reservoirs[idx] = R;
    //lights.samples[idx].out_radiance = out_radiance / f32(SAMPLES);
    //lights.samples[idx].W = lights.samples[idx].w / max(0.00001, (f32(unpack4xU8(lights.samples[idx].M_valid).x) * p_hat(sam_from_res(lights.samples[idx]), pixel_normal, lights.samples[idx].sample_point, pixel_albedo, -normalize(lights.samples[idx].sample_point - cam_loc))));
    //textureStore(output, id.xy, vec4<f32>(vec3((lights.samples[idx].W / 5.0)), 1.0));
}

struct SampleReturn {
    color: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    point: vec3<f32>,
    emission: vec3<f32>,
    visible_point: vec3<f32>,
    visible_normal: vec3<f32>,
    out_radiance: vec3<f32>,
    ty:u32,
    roughness: f32,
    valid: bool,
    pdf:f32,
}

struct DiffuseReturn {
    brightness: vec3<f32>,
    direction: vec3<f32>,
    dot: f32,
}

struct PickedRefIdx {
    picked_idx: f32,
}

struct RISDirections {
    dir: vec3<f32>,
    weight: f32,
}

fn rt_sample(coord: vec2<u32>, own_seed: u32, should_push:bool) -> SampleReturn {
    var self_seed = own_seed;
    var ray = create_ray(coord);
    // add to sample when hitting a light source
    var color: vec3<f32> = vec3<f32>(1.0);
    var out_radiance_colour: vec3<f32> = vec3<f32>(1.0);
    var sample: SampleReturn = SampleReturn();
    sample.ty = SKY;
    camera_loc = ray.origin;
    sample.pdf = 1.0;
    var hit_state = HitState();
    var previous_brightness = 0.0;
    // like i but only starts counting after first diffuse surface or any surface that has a roughness > 0.2
    var i_diffuse = 0u;
    var accum_emission = vec3<f32>();
    var accum_albedo = vec3<f32>(1.0);
    var picked = PickedRefIdx();
    var last_not_vmf = false;
    var vmf_weight = 1.0;
    let start_dir = ray.direction;

    let mc_chain = tentative_marcov_chain_state;

    var last_diffuse = false;
    let mul = pick_ref_idx_blend(&picked, self_seed);
    
    color *= mul;
    out_radiance_colour *= mul;

    for (var i = 0u; i < BOUNCES; i++) {
        let intersection = ray_hit(&ray);
        if (intersection.hit) {
            let new_pos = at(&ray, intersection.t);
            // We have at least hit the scene twice (this is our second time)
            if i != 0u {
                let radiance_reflected = strip_NaN_vec3((intersection.color_base * grid_load_radiance_estimate(new_pos, intersection.normal, ray.direction)) * hit_state.cached_brdf);
                let radiance_out = (intersection.emission * hit_state.cached_brdf);
                grid_add_radiance(ray.origin, hit_state.last_normal, hit_state.last_dir, (radiance_out + radiance_reflected) / hit_state.cached_pdf);
                let weight = length((radiance_reflected) + radiance_out);
                if weight > 0.0 {
                    let state = update_marcov_state(ray.origin, new_pos, weight, self_seed, false, grid_load(ray.origin, hit_state.last_normal, hit_state.last_dir, camera_loc, false), 1.0, sum_mc, NUM_RESAMPLES);
                    grid_store_markov_chain(ray.origin, hit_state.last_normal, hit_state.last_dir, camera_loc, state);
                    self_seed++;
                }
                tentative_marcov_chain_state = update_marcov_state(ray.origin, new_pos, length(intersection.emission), self_seed, hit_state.last_not_vmf, tentative_marcov_chain_state, select(0.5, 1.0, i == 1), sum_mc, NUM_RESAMPLES);
            }
            hit_state.last_normal = intersection.normal;
            hit_state.last_dir = ray.direction;
            hit_state.last_not_vmf = false;
            hit_state.last_diffuse = false;
            ray.origin = new_pos;

            if (i == 0u) {
                sample.point = ray.origin;
                sample.ty = intersection.ty;
                //sample.roughness = intersection.roughness;
                sample.normal = intersection.normal;
            } else if (i == 1u) {
                // dbg_out = vec3(all_vmf_pdfs(camera_loc, normalize(vec3(0.5, 1.0, 0.5) - camera_loc)) / 500.0);
                // dbg_out = vec3(all_vmf_pdfs(camera_loc, normalize(vec3(0.62, 0.66, 0.44) - camera_loc)) / 100.0);
                sample.visible_point = ray.origin;
                sample.visible_normal = intersection.normal;
            }
            /*if ((intersection.tri_area > SMALL_VAL) && previously_diffuse) {
                //sample.color = vec3f(intersection.tri_area);
                //return sample;
            }*/
            /*if ((intersection.tri_area > SMALL_VAL) && should_push && previously_diffuse) {
                push(vec4<f32>(ray.origin, intersection.tri_area));
                previously_diffuse = false;
                //sample.color = vec3f(intersection.tri_area);
                //return sample;
            }*/
            
            let hit = handle_hit(&hit_state, i, intersection, &ray, &sample, &picked, &self_seed);
            
            sample.color = fma(color, hit.emission, sample.color);
            if (!(i == 0u)) {
                sample.out_radiance = fma(out_radiance_colour, hit.emission, sample.out_radiance);
            }

            // sample.color = vec3(hit.albedo);
            // return sample;

            color *= hit.albedo;
            if (!(i == 0u)) {
                out_radiance_colour *= hit.albedo;
            }

            if i == 0 {
                hit_state.first_dir = ray.direction;
                sample.albedo = hit.albedo;
                sample.emission = hit.emission;
            }
            //sample.color = ray.direction;
            //return sample;
        } else {
            let bg_color = background(&ray);
            // if it's our first hit it doesn't matter, but we should resample the sky sometimes
            let new_pos = at(&ray, T_MAX);
            if (i == 1u) {
                sample.visible_point = new_pos;
                sample.visible_normal = -ray.direction;
            }
            if (!(i == 0u)) {
                sample.out_radiance = fma(out_radiance_colour, bg_color, sample.out_radiance);

                // update markov chains
                let radiance_out = (bg_color * hit_state.cached_brdf);
                grid_add_radiance(ray.origin, hit_state.last_normal, hit_state.last_dir, radiance_out / hit_state.cached_pdf);
                let weight = length(radiance_out);
                if weight > 0.0 {
                    let state = update_marcov_state(ray.origin, new_pos, weight, self_seed, false, grid_load(ray.origin, hit_state.last_normal, hit_state.last_dir, camera_loc, false), 1.0, sum_mc, NUM_RESAMPLES);
                    grid_store_markov_chain(ray.origin, hit_state.last_normal, hit_state.last_dir, camera_loc, state);
                    tentative_marcov_chain_state = update_marcov_state(ray.origin, new_pos, length(radiance_out), self_seed, false, tentative_marcov_chain_state, select(0.5, 1.0, i == 1), sum_mc, NUM_RESAMPLES);
                    self_seed++;
                }
            }
            sample.color = fma(color, bg_color, sample.color);
            break;
        }
    }

    // A final train of the markov chain with the real radiance
    //sample.out_radiance = (sample.color - sample.emission) / sample.albedo;
    let world_cache = grid_load(sample.point, sample.normal, start_dir, camera_loc, false);
    let state = update_marcov_state(sample.point, sample.visible_point, length(sample.out_radiance) * hit_state.first_pdf, self_seed, false, world_cache, 1.0, sum_mc, NUM_RESAMPLES);
    grid_store_markov_chain(sample.point, sample.normal, start_dir, camera_loc, state);
    // Sometimes the radiance cache can be a little off (especially on reflections), this fixes it.
    grid_add_radiance(sample.point, sample.normal, start_dir, sample.out_radiance);

    // tentative_marcov_chain_state = update_marcov_state(sample.point, sample.visible_point, length(sample.out_radiance), self_seed, false, tentative_marcov_chain_state, 0.1);

    sample.color /= vmf_weight;
    sample.out_radiance /= vmf_weight;
    storageBarrier();
    // sample.color = dbg_out;
    // sample.albedo = first_dir;
    // sample.color = vec3(get_kappa(tentative_marcov_chain_state) / 100.0);
    // sample.color = vec3((normalize(get_light_dir(mc_chain, sample.point)) + 1.0) / 2.0);
    // sample.color = vec3(get_kappa(world_cache) / 5.0);
    // sample.color = vec3(all_vmf_pdfs(camera_loc, first_dir));
    // sample.color = vec3(all_vmf_pdfs(camera_loc, first_dir));
    // sample.color = vec3(all_vmf_pdfs(camera_loc, normalize(vec3(0.62, 0.66, 0.44) - camera_loc)));
    // sample.color = vec3(all_vmf_pdfs(camera_loc, normalize(vec3(0.5, 1.0, 0.5) - camera_loc)) / 500.0);
    // sample.color = vec3(vmf_pdf(normalize(get_light_dir(tentative_marcov_chain_state, camera_loc)), get_kappa(tentative_marcov_chain_state), first_dir) / 5.0);
    // sample.color = vec3((normalize(get_light_dir(world_cache, sample.point)) + 1.0) / 2.0);
    //sample.color = safe_div_vec3(state.light_source, state.weight_sum);
    //sample.color = vec3(f32(world_cache.num_samples) / f32(MARCOV_CHAIN_SAMPLE_CLAMP), (f32(world_cache.num_samples) / f32(MARCOV_CHAIN_SAMPLE_CLAMP)) * 10.0, (f32(world_cache.num_samples) / f32(MARCOV_CHAIN_SAMPLE_CLAMP)) * 100.0);
    //sample.color = vec3(world_cache.weight_sum / 1.0);
    //sample.color = sample.out_radiance;
    // sample.color = (grid_load_radiance_estimate(sample.point, sample.normal, start_dir));
    let grid_pos = vec3<i32>(sample.point / GRID_SIZE);
    let hash = primary_hasher(grid_pos, sample.normal, start_dir);
    //sample.color = u32_to_colour(hash);
    //let tex_coord = vec2<u32>((vec2<f32>(coord) / vec2<f32>(textureDimensions(output))) * 64.0);
    //sample.color = textureLoad(tex_attributes, tex_coord, 0u, 0).xyz;
    //sample.color = vec3<f32>(vec2<f32>(tex_sizes[1u].size[1]), 0.0);
    //sample.pdf = saturate(sample.pdf);
    //sample.color = strip_NaN_vec3(sample.color);
    sample.valid = true;//(i_diffuse > 1u);
    return sample;
}

struct FinalizedHit {
    albedo: vec3<f32>,
    emission: vec3<f32>,
}

fn FinalizedHit_new() -> FinalizedHit {
    return FinalizedHit(vec3(1.0), vec3(0.0));
}

struct RefractionChecks {
    cos: f32,
    cannot_refract: bool,
}

fn check_refract(dir: vec3<f32>, normal: vec3<f32>, refractive_index: f32) -> RefractionChecks {
    let cos_theta = min(dot(-dir, normal), 1.0);
    let sin_theta = sqrt(1.0 - (cos_theta*cos_theta));
    let cannot_refract = (refractive_index * sin_theta) > 1.0;
    return RefractionChecks(cos_theta, cannot_refract);
}

// The multibounce hit state.
//
// A collection of various things that can be zeroed.
struct HitState {
    first_dir: vec3<f32>,
    last_normal: vec3<f32>,
    cached_pdf: f32,
    cached_brdf: f32,
    last_dir: vec3<f32>,
    first_pdf: f32,
    last_not_vmf: bool,
    last_diffuse: bool,
    previously_diffuse: bool,
}

fn handle_hit(
    state: ptr<function, HitState>,
    bounce_idx: u32,
    intersection: Intersection,
    ray: ptr<function, Ray>,
    sample: ptr<function, SampleReturn>,
    picked: ptr<function, PickedRefIdx>,
    self_seed: ptr<function, u32>,
) -> FinalizedHit {
    var roughness = intersection.roughness_specular;
    var hit = FinalizedHit_new();
    
    // coated-base

    // First, possibly skip this
    var skip_coated_base: bool;
    {
        let rand = rand_f32_ptr(self_seed);
        skip_coated_base = rand > intersection.color_coat.w;
    }

    var to_base_substrate = true;
    let colour_refractive_index = pick_ref_idx(picked, intersection.refractive_index_low, intersection.refractive_index_high, *self_seed);
    var refractive_index = colour_refractive_index;
    *self_seed = *self_seed + 1u;
    if (!intersection.front_face && intersection.ty == TRANSPARENT) {
        // disable the coat reflection if we are exiting a coated transparent
        refractive_index = 1.0;
    } else {
        refractive_index = 1.0 / refractive_index;
    }

    let coat_normal: PickedNormal = get_normal((*ray).direction, intersection.normal, intersection.tangent, *self_seed, roughness);
    // let onb = to_onb((*ray).direction, intersection.normal, intersection.tangent);
    // hit.albedo = vec3(((onb + 1.0) / 2.0).y);
    // return hit;
    var reflect_normal = coat_normal;

    if (refractive_index != 1.0) {
        let checks = check_refract((*ray).direction, coat_normal.normal, refractive_index);
        // REMEMBER: Looking from inside a material looks odd, but this seems to be correct.
        let pdf = reflectance(checks.cos, refractive_index);
        let rand = rand_f32_ptr(self_seed);
        if ((checks.cannot_refract || (pdf > rand))) {
            roughness = intersection.roughness_coat;
            to_base_substrate = false;
        }
    }

    to_base_substrate = to_base_substrate || skip_coated_base;

    if to_base_substrate {
        hit.albedo *= mix(vec3(1.0), intersection.color_coat.xyz, intersection.color_coat.w);
        hit.emission = fma(intersection.emission, hit.albedo, hit.emission);
    }

    // Mostly checking if base-substrate is metal, but also a bit of coated base
    // to prevent quite so many branches

    var is_metal = intersection.ty == METALLIC;

    {
        let rand = rand_f32_ptr(self_seed);
        is_metal = is_metal || (rand < intersection.metalness);
    }

    // Don't want to be metal if we should be reflecting off the coat.
    is_metal = is_metal && to_base_substrate;

    var specular_refractive_index = pick_ref_idx(picked, intersection.specular_refractive_index_low, intersection.specular_refractive_index_high, *self_seed);
    *self_seed = *self_seed + 1u;
    // Don't have to check if it is thin walled as front_face is always true in this case.
    if (!intersection.front_face && intersection.ty == TRANSPARENT) {
        specular_refractive_index = specular_refractive_index;
    } else {
        specular_refractive_index = 1.0 / specular_refractive_index;
    }
    
    let specular_normal: PickedNormal = get_normal((*ray).direction, intersection.normal, intersection.tangent, *self_seed, intersection.roughness_specular);

    var to_base = true;
    if is_metal {
        reflect_normal = specular_normal;
        let F_0 = intersection.color_base.xyz;
        let cos: f32 = min(dot(-(*ray).direction, reflect_normal.normal), 1.0);
        let one_minus_cos = 1.0 - cos;
        let F_Schlick = F_0 + ((1.0 - F_0) * pow(one_minus_cos, 5.0));
        const REWEIGHT_COS = 1.0 / 7.0;
        const REWEIGHT = REWEIGHT_COS * pow(1.0 - REWEIGHT_COS, 6.0);
        const REWEIGHT_COS_Pow5 = pow(1.0 - REWEIGHT_COS, 5.0);
        let F_Schlick_1_7 = F_0 + ((1.0 - F_0) * REWEIGHT_COS_Pow5);
        let F = intersection.color_specular.xyz * F_Schlick_1_7;
        let F_82 = F_Schlick - (((cos * pow(1.0 - cos, 6.0)) / REWEIGHT) * (F_Schlick_1_7 - F));

        hit.albedo *= F_82 * intersection.color_specular.w;
    } else {    
        if (specular_refractive_index != 1.0) {
            let checks = check_refract((*ray).direction, specular_normal.normal, specular_refractive_index);
            // REMEMBER: Looking from inside a material looks odd, but this seems to be correct.
            let pdf = reflectance(checks.cos, specular_refractive_index);
            let rand = rand_f32_ptr(self_seed);
            if ((checks.cannot_refract || (pdf > rand))) {
                reflect_normal = specular_normal;
                to_base = false;
                hit.albedo *= intersection.color_specular.xyz;
            }
        }
    }

    if is_metal || !to_base_substrate || !to_base {
        (*state).cached_pdf = reflect_normal.pdf;
        if bounce_idx == 0u {
            (*state).first_pdf = (*state).cached_pdf;
        }
        // hit.albedo = vec3(f32(dot((*ray).direction, from_onb(to_onb((*ray).direction, intersection.normal, intersection.tangent), intersection.normal, intersection.tangent)) > 0.99));
        // let dot_prod = dot(-(*ray).direction, reflect_normal.normal);
        // let dot_prod = reflect_normal.onb_normal.y;
        // let dot_prod = dot(reflect_normal.onb_normal, to_onb(-(*ray).direction, intersection.normal, intersection.tangent));
        // hit.albedo = vec3(f32(dot_prod > 0.0), dot_prod, f32(-dot_prod > 0.2));
        // return hit;
        let old_dir = (*ray).direction;
        (*ray).direction = reflect((*ray).direction, reflect_normal.normal);
        (*state).cached_brdf = get_normal_bsdf(old_dir, (*ray).direction, reflect_normal, intersection.normal, intersection.tangent);
        hit.albedo *= (*state).cached_brdf / (*state).cached_pdf;
        if dot((*ray).direction, intersection.normal) < 0.0 {
            hit.albedo = vec3(0.0);
        }
        return hit;
    }

    fill_mc_world_space_states(self_seed, (*ray).origin, intersection.normal, (*ray).direction);
    var s = resample_mcmm(self_seed, (*ray).origin, intersection.normal);

    //sample.color = (normal);
    //return sample;
    //sample.color = color;
    //return sample;

    var use_subsurface: bool;
    var use_subsurface_back: bool;
    {
        let rand = rand_f32_ptr(self_seed);
        use_subsurface = rand < intersection.subsurface_weight;
        let rand_back = rand_f32_ptr(self_seed);
        use_subsurface_back = (((1.0 + intersection.subsurface_anisotropy) / 2) < rand_back) && use_subsurface;
    }

    hit.albedo *= select(intersection.color_base.xyz, intersection.color_subsurface.xyz, use_subsurface);

    if (intersection.ty == DIFFUSE) {
        (*state).last_diffuse = true;
        // old direction code
        /*let direction = rand_on_sphere(self_seed);
        let dot = dot(direction, intersection.normal);
        color = color * abs(dot);
        if (dot < 0.0) {
            ray.direction = -direction;
        } else {
            ray.direction = direction;
        }*/
        var dir: vec3<f32>;
        // var wrs = weighted_reservoir_sampling::WRS_vec4f();
        // Prepare for markov chain
        //s.light_source = vec3<f32>(0.0, 1.0, 0.0);
        let light_dir = get_light_dir(s, (*ray).origin);
        let mean = normalize(light_dir);
        let kappa = get_kappa(s);

        let normal = select(intersection.normal, -intersection.normal, use_subsurface_back);
        // hit.albedo = vec3(f32(use_subsurface_back));
        // return hit;
        // Calculate cosine direction
        let dir_cosine = cosine_weighted_hemisphere(*self_seed, normal, intersection.tangent);
        let cosine_pdf = max(dot(dir_cosine, normal) / PI, 0.0);
        // let cosine_weight = 0.2;
        // weighted_reservoir_sampling::add_to_wrs_vec4f(&wrs, vec4(dir_cosine, cosine_pdf), cosine_weight, &self_seed);
        // Calculate markov chain direction
        let theta = rand_f32_ptr(self_seed) * TAU;
        let V = vec2<f32>(cos(theta), sin(theta));
        let xi = rand_f32_ptr(self_seed);
        let minus_2_kappa = -2.0 * kappa;
        let W_minus_1 = ((log(1.0 + (xi * exp_minus_one(minus_2_kappa))) / kappa));
        var W = 1.0 + W_minus_1;
        // slightly more precise than `sqrt(1.0 - (W*W))`
        let norm_V = sqrt(-fma(W_minus_1, W_minus_1, 2*W_minus_1));
        let Xs = from_onb_no_tangent(vec3(strip_NaN(V.x * norm_V), W, strip_NaN(V.y * norm_V)), mean);
        let dir_markov = normalize(Xs);
        // Calculate pdf of cosine for cosine direction
        var unnormalied_confidences = array<f32, 2>(0.5, 0.5);
        //var unnormalied_confidences = array<f32, 2>(0.8 , 0.2);
        if (!marcov_chain_valid(s) || near_zero(length(light_dir)) || near_zero(s.weight_sum) || near_zero(length(Xs))) {
            unnormalied_confidences[0] = 1.0;
            unnormalied_confidences[1] = 0.0;
        }
        let confidence_sum = unnormalied_confidences[0] + unnormalied_confidences[1];
        var confidences = array<f32, 2>(unnormalied_confidences[0] / confidence_sum, unnormalied_confidences[1] / confidence_sum);

        // var markov_weight = confidences[1];
        // if dot(dir_markov, intersection.normal) < 0.0 {
        //     //markov_weight = 0.0;
        // }
        // let markov_chain_pdf = select(all_vmf_pdfs(ray.origin, dir), 0.0, confidences[1] == 0.0);
        // weighted_reservoir_sampling::add_to_wrs_vec4f(&wrs, vec4(dir_markov, markov_chain_pdf), markov_weight, &self_seed);
        var final_pdf: f32;
        var norm_divisor = 1.0;
        let weight = confidences[0];
        let rand = rand_f32_ptr(self_seed);
        if (rand < weight) {
            dir = dir_cosine;
            //final_pdf = max(dot(dir, intersection.normal) / PI, 0.0);
            (*state).last_not_vmf = true;
        } else {
            dir = (dir_markov);
            //final_pdf = vmf_pdf(mean, kappa, dir_markov);
            //sample.color = (dir_markov + 1) / 2;
            //return sample;
        }
        let avg_pdf = ((confidences[0] * cosine_pdf) + select(confidences[1] * max(all_vmf_pdfs((*ray).origin, dir), 0.0), 0.0, confidences[1] == 0.0));

        // let avg_pdf = (wrs.current_weight / wrs.weight_sum) * wrs.current_elem.w;
        // let avg_pdf = cosine_pdf * (cosine_weight / wrs.weight_sum) + markov_chain_pdf * (markov_weight / wrs.weight_sum);
        // dir = wrs.current_elem.xyz;
        final_pdf = avg_pdf;
        norm_divisor = avg_pdf;
        //vmf_weight *= avg_pdf;
        //final_pdf = 1.0;
        // Calculate final pdf
        let diffuse: vec3<f32> = select(make_diffuse(intersection.color_base, (*ray), intersection.normal, intersection.tangent, intersection.roughness.x, dir), vec3(1.0), use_subsurface);
        //let dot = max(dot(dir, normal), 0.0) / dot(dir, intersection.normal);
        //color = color * dot;
        //ray.direction = dir;
        if (any(strip_NaN_vec3(dir) != dir)) {
            //sample.color = vec3f(0.0, 1.0, 0.0);
            //return sample;
        }
        (*state).cached_pdf = final_pdf;
        if bounce_idx == 0u {
            (*state).first_pdf = (*state).cached_pdf;
        }
        (*state).cached_brdf = max(abs(dot((dir), intersection.normal)) / PI, 0.0);
        let normalization_factor = ((*state).cached_brdf) / max(norm_divisor, 0.001);
        hit.albedo *= normalization_factor * diffuse;

        (*ray).direction = dir;
        (*state).previously_diffuse = true;
        if (bounce_idx == 0) {
            (*sample).pdf = final_pdf;
        }
        //sample.color = vec3f((diffuse.w));
        //sample.color = ray.direction;
        //return sample;
    } else if (intersection.ty == TRANSPARENT) {
        (*state).cached_pdf = 1.0;
        (*state).cached_brdf = 1.0;
        if bounce_idx == 0u {
            (*state).first_pdf = (*state).cached_pdf;
        }
        //let k = 1.0 - refractive_index * refractive_index * (1.0 - dot(intersection.normal, ray.direction) * dot(intersection.normal, ray.direction));
        if !intersection.thin_walled { (*ray).direction = refract((*ray).direction, specular_normal.normal, specular_refractive_index); }
        let dot = dot((*ray).direction, intersection.normal);

        if dot > 0.0 {
            hit.albedo = vec3(0.0);
        }
        //sample.color = vec3f(sign(k), -sign(k), sign(dot(intersection.normal, ray.direction)));
        //sample.color = vec3f((refractive_index * refractive_index) / 8.0, (1.0 - dot(intersection.normal, ray.direction) * dot(intersection.normal, ray.direction)), f32((1.0 - dot(intersection.normal, ray.direction) * dot(intersection.normal, ray.direction)) > 0.25));
        //sample.color = -normal;
        //return sample;
    }
    return hit;
}

fn u32_to_colour(u:u32) -> vec3<f32> {
    let r = (0xFFE00000 & u) >> 20;
    let g = (0xFFC00 & u) >> 10;
    let b = (0x3FF & u);
    return vec3<f32>(f32(r) / f32(0x7FF), f32(g) / f32(0x3FF), f32(b) / f32(0x3FF));
}

fn all_vmf_pdfs(hit:vec3<f32>, direction:vec3<f32>) -> f32 {
    var pdfs = 0.0;
    if (sum_mc > 0.0) {
        for (var i = 0; i < NUM_RESAMPLES; i++) {
            let s = other_pixels_marcov_chain_states.state[i];
            if (!marcov_chain_valid(s)) {
                continue;
            }
            pdfs += (vmf_pdf(normalize(get_light_dir(s, hit)), get_kappa(s), direction)) * (s.weight_sum / sum_mc);
            //pdfs += vmf_pdf(normalize(s.light_source - hit), get_kappa(s), direction) / f32(NUM_RESAMPLES);
            //pdfs += 1.0;
        }
    }
    return pdfs;
}

fn get_kappa(state: MarcovChainState) -> f32 {
    const Np = 0.2;
    // TODO: use rp mix, that was the best in the paper.
    let rp = 0.99;
    // Not exactly sure what N is supposed to be in this case, but pretty sure it is this.
    let N = state.num_samples;
    let N_sqrd = f32(N * N);
    //let r = safe_div(((N_sqrd * (0.99)) + (Np * rp)), (N_sqrd + Np));
    let r = (((N_sqrd * (state.mean_cosine / state.weight_sum)) + (Np * rp)) / (N_sqrd + Np));
    //let r = rp;
    // We don't want this to be zero or infinity
    // These values are large due to the large amount of floating point inacuracies there are
    //return max((3.0 * r) - (r*r*r) / max(1 - (r*r), 1.17549435E-38f), 1.17549435E-38f);
    return clamp(((3.0 * r) - (r*r*r)) / (1.0 - (r*r)), 0.001, 8.50705867e37);
    //return (((3.0 * r) - (r*r*r)) / (1.0 - (r*r)));
    //return 1000.0;
}

fn vmf_pdf(mean:vec3<f32>, kappa:f32, direction:vec3<f32>) -> f32 {
    let minus_2_kappa = -2.0 * kappa;
    let fraction = x_divided_by_exp_minus_one(minus_2_kappa) / (4.0 * PI);
    let cos_theta = dot(direction, mean);
    return max(fraction*exp(kappa * (cos_theta - 1.0)), 0.0);
}

fn exp_minus_one(x: f32) -> f32 {
    let u = exp(x);
    if (u == 1.0) {
        return x;
    }
    let y = u - 1.0;
    if (abs(x) < 1.0) {
        return (x * y) / log(u);
    }
    return y;
}

fn x_divided_by_exp_minus_one(x:f32) -> f32 {
    let u = exp(x);
    if (u == 1.0) {
        return 1.0;
    }
    let y = u - 1.0;
    if (abs(x) < 1.0) {
        return log(u) / y;
    }
    return x / y;
}

override T_MIN:f32;
override T_MAX:f32;


const BOUNCES = 5u;
const VERTICES = BOUNCES + 1u;

struct Ray {
    direction: vec3f,
    origin: vec3f,
}

struct Intersection {
    color_base: vec3f,
    hit: bool,
    emission: vec3f,
    // Always true if thin walled is on - both sides are the "front"
    front_face: bool,
    color_coat: vec4f,
    color_subsurface: vec3f,
    ty: u32,
    normal: vec3f,
    thin_walled: bool,
    tangent: vec3<f32>,
    subsurface_anisotropy: f32,
    roughness: vec2<f32>,
    roughness_coat: vec2<f32>,
    refractive_index_low: f32,
    refractive_index_high: f32,
    specular_refractive_index_low: f32,
    specular_refractive_index_high: f32,
    subsurface_weight: f32,
    t: f32,
    metalness: f32,
    roughness_specular: vec2<f32>,
    color_specular: vec4f,
}

fn ray_hit(ray: ptr<function, Ray>) -> Intersection {
    var rq: ray_query_maybe_vertex_return;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, T_MIN, T_MAX, (*ray).origin, (*ray).direction));
    var aabb_normal: vec3<f32>;
    var aabb_tangent: vec3<f32>;
    while (rayQueryProceed(&rq)) {
        let intersection = rayQueryGetCandidateIntersection(&rq);
        if (intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
            let material = materials[material_idx[intersection.instance_index].materials[intersection.primitive_index][0]];
            let texture_pos = tex_pos[material_idx[intersection.instance_index].materials[intersection.primitive_index][1]];
            let bary = vec3<f32>(intersection.barycentrics, ((1.0 - intersection.barycentrics.x) - intersection.barycentrics.y));
            let tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_base_colour_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_base_colour_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_base_colour_3) * bary.y)));
            let idx_diffuse = unpack_2xu16(material.tex_idx_base_colour_emission).x;
            let ty = unpack4xU8(material.ty_thin_walled_subsurface).x;
            if (!near_zero(textureSampleLevel(textures[idx_diffuse], sam, tex_coords_float, intersection.t).w) || ty == TRANSPARENT) {
                rayQueryConfirmIntersection(&rq);
            }
        } else if (intersection.kind == RAY_QUERY_INTERSECTION_AABB) {
            let aabb_intersection = intersect(intersection);
            if (aabb_intersection.hit) {
                aabb_normal = aabb_intersection.normal;
                aabb_tangent = aabb_intersection.tangent;
                let t = aabb_intersection.t;
                rayQueryGenerateIntersection(&rq, t);
            }
        }
    }
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var return_intersection = Intersection();
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        return_intersection.hit = true;
        return_intersection.t = intersection.t;
        return_intersection.front_face = intersection.front_face;

        let material = materials[material_idx[intersection.instance_index].materials[intersection.primitive_index][0]];
        let texture_pos = tex_pos[material_idx[intersection.instance_index].materials[intersection.primitive_index][1]];
        let idx_diffuse_emission = unpack_2xu16(material.tex_idx_base_colour_emission);
        let idx_roughness_coat = unpack_2xu16(material.tex_idx_roughness_coat);
        let idx_subsurface_metalness = unpack_2xu16(material.tex_idx_subsurface_metalness);
        let idx_specular = unpack_2xu16(material.tex_idx_specular_colour_specular_rougness);
        let ty_thin_walled_subsurface = unpack4xU8(material.ty_thin_walled_subsurface);
        return_intersection.ty = ty_thin_walled_subsurface.x;
        return_intersection.thin_walled = bool(ty_thin_walled_subsurface.y & (1 << 7));

        if return_intersection.thin_walled {
            return_intersection.front_face = true;
        }

        const SUBSURFACE_MASK = (1 << 7) - 1;
        return_intersection.subsurface_weight = f32(ty_thin_walled_subsurface.y & (SUBSURFACE_MASK)) / f32(SUBSURFACE_MASK);
        return_intersection.subsurface_anisotropy = unpack2x16snorm(material.ty_thin_walled_subsurface).y;
        let ref_idx = unpack2x16float(material.refractive_index);
        return_intersection.refractive_index_low = ref_idx.x;
        return_intersection.refractive_index_high = ref_idx.y;
        let ref_idx_spec = unpack2x16float(material.specular_refractive_index);
        return_intersection.specular_refractive_index_low = ref_idx_spec.x;
        return_intersection.specular_refractive_index_high = ref_idx_spec.y;

        var normal: vec3<f32>;
        var tangent: vec3<f32>;
        if (intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
            var vertices: array<vec3<f32>, 3>;
            @if(vertex_return) {
                vertices = getCommittedVertexPositions(&rq);
            }
            @if(!vertex_return) {
                vertices = get_vertices(rayQueryGetCommittedIntersection(&rq));
            }
            tangent = correct_tangent(normalize(vertices[0] - vertices[1]));
            normal = normalize(cross(tangent, vertices[0] - vertices[2]));
        } else if (intersection.kind == RAY_QUERY_INTERSECTION_GENERATED) {
            tangent = aabb_tangent;
            normal = aabb_normal;
        }

        if (dot((*ray).direction, normal) > 0.0) {
            return_intersection.normal = -normal;
        } else {
            return_intersection.normal = normal;
        }
        return_intersection.tangent = tangent;

        let bary = vec3<f32>(intersection.barycentrics, ((1.0 - intersection.barycentrics.x) - intersection.barycentrics.y));
        let colour_tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_base_colour_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_base_colour_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_base_colour_3) * bary.y)));
        let emission_tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_emission_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_emission_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_emission_3) * bary.y)));
        let attributes_tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_attributes_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_attributes_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_attributes_3) * bary.y)));
        let coat_tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_coat_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_coat_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_coat_3) * bary.y)));
        let sub_surface_tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_sub_surface_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_sub_surface_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_sub_surface_3) * bary.y)));
        let metalness_tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_metalness_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_metalness_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_metalness_3) * bary.y)));
        let specular_tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_specular_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_specular_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_specular_3) * bary.y)));
        
        //return_intersection.color = vec3f(tex_coords_float, 0.0);
        let recolour = textureLoad(tex_recolour, unpack_2xu16(material.tex_pos_recolour), 0).xyz;
        let diffuse_colour = textureSampleLevel(textures[idx_diffuse_emission.x], sam, colour_tex_coords_float, intersection.t).xyz * recolour;
        return_intersection.color_base = diffuse_colour;
        if (idx_roughness_coat.y != 0xFFFFu) {
            let coat_colour = textureSampleLevel(textures[idx_roughness_coat.y], sam, coat_tex_coords_float, intersection.t);
            return_intersection.color_coat = vec4(coat_colour.xyz * recolour, coat_colour.w);
        } else {
            return_intersection.color_coat = vec4(1.0, 1.0, 1.0, 0.0);
        }
        
        if (idx_subsurface_metalness.x != 0xFFFFu) {
            let subsurface_colour = textureSampleLevel(textures[idx_subsurface_metalness.x], sam, sub_surface_tex_coords_float, intersection.t).xyz * recolour;
            return_intersection.color_subsurface = subsurface_colour;
        } else {
            return_intersection.color_subsurface = vec3(0.8, 0.8, 0.8);
        }
        if (idx_subsurface_metalness.y != 0xFFFFu) {
            let metalness = textureSampleLevel(textures[idx_subsurface_metalness.y], sam, metalness_tex_coords_float, intersection.t).x;
            return_intersection.metalness = metalness;
        } else {
            return_intersection.metalness = 0.0;
        }
        let emission_scale = bitcast<vec2<f32>>(unpack_2xu16(material.emission_scale) << vec2(16));
        if (idx_diffuse_emission.y != 0xFFFFu) {
            return_intersection.emission = textureSampleLevel(textures[idx_diffuse_emission.y], sam, emission_tex_coords_float, intersection.t).xyz * emission_scale.x;
        }
        return_intersection.emission += diffuse_colour * emission_scale.y;
        if (idx_roughness_coat.x != 0xFFFFu) {
            let attributes = textureSampleLevel(textures[idx_roughness_coat.x], sam, attributes_tex_coords_float, intersection.t);
            return_intersection.roughness = attributes.xy;
            return_intersection.roughness_coat = attributes.zw;
        }
        if (idx_specular.x != 0xFFFFu) {
            let specular_colour = textureSampleLevel(textures[idx_specular.x], sam, specular_tex_coords_float, intersection.t);
            return_intersection.color_specular = specular_colour;
        } else {
            return_intersection.color_specular = vec4(1.0);
        }
        if (idx_specular.y != 0xFFFFu) {
            let specular_rougness = textureSampleLevel(textures[idx_specular.y], sam, specular_tex_coords_float, intersection.t).xy;
            return_intersection.roughness_specular = specular_rougness;
        } else {
            return_intersection.roughness_specular = vec2(0.0);
        }
    }
    return return_intersection;
}

fn correct_tangent(tangent: vec3<f32>) -> vec3<f32> {
    var dir = tangent;
    let abs_dir = abs(dir);
    let x_largest = abs_dir.x > abs_dir.y && abs_dir.x > abs_dir.z;
    let y_largest = abs_dir.y > abs_dir.x && abs_dir.y > abs_dir.z;
    var reverse_dir: bool;
    if (x_largest) {
        reverse_dir = dir.x < 0.0;
    } else if (y_largest) {
        reverse_dir = dir.y < 0.0;
    } else {
        reverse_dir = dir.z < 0.0;
    }
    if (reverse_dir) {
        dir = -dir;
    }
    return dir;
}

fn create_ray(coord: vec2<u32>) -> Ray {
    let screen_size = textureDimensions(output);
    let half_size = (vec2<f32>(screen_size) / 2.0);
    //the coordinates get converted from screenspace to between -1 and 1
    let normalized_coords = (vec2<f32>(coord) / half_size) - 1.0;
    // textures start at the top and go down so reverse
    let reverse_y_coords = vec2<f32>(normalized_coords.x, -normalized_coords.y);
    // since normal projection projects from world coords (rotated and translated already) to -1 and 1
    // inverting it will convert from [-1, 1] to world coords (currently still rotated, translated)
    let projected_coords = normalize(camera.projection_inverse * vec4<f32>(reverse_y_coords, 1.0, 1.0));
    // then we rotate these coordinates, but since the w location controls translation scale this will not translate
    let direction = (camera.view_inverse * vec4<f32>(projected_coords.xyz, 0.0)).xyz;
    // the position 0.0 gets rotated to 0.0, so only the tranlation gets applied
    let position = (camera.view_inverse * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    return Ray(direction, position);
}

// get the location at a distance down the ray
fn at(ray: ptr<function, Ray>, t: f32) -> vec3<f32> {
    return fma((*ray).direction, vec3<f32>(t), (*ray).origin);
}

fn background(ray: ptr<function, Ray>) -> vec3<f32> {
    return textureSampleLevel(bg, sam, (*ray).direction, 0.0).xyz;
}

// from https://math.stackexchange.com/a/1586185
// also see https://mathworld.wolfram.com/SpherePointPicking.html
fn rand_on_sphere(own_seed:u32) -> vec3<f32> {
    let u1 = rand_f32(own_seed);
    let u2 = rand_f32(own_seed + 1u);
    let latitude = acos(fma(2.0, u1, -1.0)) - HALF_PI;
    let longitude = TAU * u2;
    return to_sphere_from_lat_long(latitude, longitude);
}

fn rand_on_hemisphere(own_seed:u32) -> vec3<f32> {
    let u1 = rand_f32(own_seed);
    let u2 = rand_f32(own_seed + 1u);
    let latitude = acos(fma(2.0, u1, -1.0)) - HALF_PI;
    let longitude = TAU * u2;
    return vec3<f32>(cos(latitude) * cos(longitude), abs(sin(latitude)), cos(latitude) * sin(longitude));
}

fn to_sphere_from_lat_long(latitude: f32, longitude: f32) -> vec3<f32> {
    return vec3<f32>(cos(latitude) * cos(longitude), cos(latitude) * sin(longitude), sin(latitude));
}

fn reflectance(cosine:f32, ref_idx:f32) -> f32 {
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0_squared = r0*r0;
    return fma((1.0 - r0_squared), pow((1.0 - cosine), 5.0), r0_squared);
}

// converts from a orthonormal basis to Carteasian coordinates
fn from_onb(src:vec3<f32>, up:vec3<f32>, tangent: vec3<f32>) -> vec3<f32> {
    let s = tangent;
    let t = cross(up, s);
    return fma(vec3<f32>(src.x), s, fma(vec3<f32>(src.y), up, (src.z * t)));
}

fn from_onb_no_tangent(src:vec3<f32>, up:vec3<f32>) -> vec3<f32> {
    var other_angle: vec3<f32>;
    if (near_zero(up.x)) {
        other_angle = vec3(1.0, 0.0, 0.0);
    } else {
        other_angle = vec3(0.0, 1.0, 0.0);
    }
    let tangent = normalize(cross(up, other_angle));
    return from_onb(src, up, tangent);
}

fn onb_cosine_weighted_hemisphere(own_seed:u32) -> vec3<f32> {
    let r1 = rand_f32(own_seed);
    let r2 = rand_f32(own_seed + 1u);

    let phi = TAU * r1;
    let x = cos(phi) * sqrt(r2);
    let z = sin(phi) * sqrt(r2);
    let y = sqrt(1.0 - r2);
    return vec3<f32>(x, y, z);
}

fn cosine_weighted_hemisphere(own_seed:u32, normal: vec3<f32>, tangent:vec3<f32>) -> vec3<f32> {
    let hemisphere = onb_cosine_weighted_hemisphere(own_seed);
    return from_onb(hemisphere, normal, tangent);
}

struct PickedNormal {
    normal: vec3<f32>,
    pdf: f32,
    onb_normal: vec3<f32>,
    roughness: vec2<f32>
}

struct GGXReturn {
    normal: vec3<f32>,
    pdf: f32,
}

/// `in` is the incoming direction of the ray, it should have a -y value
fn sample_vndf_ggx(own_seed:u32, in: vec3<f32>, roughness:vec2<f32>) -> GGXReturn {
    let u1 = rand_f32(own_seed);
    let u2 = rand_f32(own_seed + 1u);

    let view_dir = -in;

    var in_direction = vec3<f32>(view_dir.x * roughness.x, view_dir.y, view_dir.z * roughness.y);

    // minimize floating point error
    if (vec3_near_zero(in_direction)) {
        in_direction = vec3<f32>(0.0, -1.0, 0.0);
    }

    let warped_dir = normalize(in_direction);

    let wm_std = sample_ggx_interal(u1, u2, warped_dir);

    var out_normal = vec3<f32>(wm_std.x * roughness.x, wm_std.y, wm_std.z * roughness.y);

    if (vec3_near_zero(out_normal)) {
        out_normal = vec3<f32>(0.0, 1.0, 0.0);
    }

    let wm = normalize(out_normal);

    let pdf = ggx_visible_D(view_dir, wm, roughness);

    return GGXReturn(wm, pdf);
}

fn sample_ggx_interal(u1: f32, u2:f32, wi:vec3<f32>) -> vec3<f32> {
    let phi = TAU * u1;
    // let y = fma((1.0 - u2), (1.0 + wi.y), -wi.y);
    let y = mix(-wi.y, 1.0, u2);
    let sin_theta = sqrt(saturate(fma(y, -y, 1.0)));
    let x = sin_theta * cos(phi);
    let z = sin_theta * sin(phi);
    let c = vec3<f32>(x, y, z);

    return c + wi;
}

/// takes an orthonormal basis
fn ggx_D(picked_normal: vec3<f32>, roughness: vec2<f32>) -> f32 {
    let heviside = select(0.0, 1.0, picked_normal.y > 0.0);

    let alpha_sqr = roughness * roughness;

    let x_sqr = picked_normal.x * picked_normal.x;
    let y_sqr = picked_normal.y * picked_normal.y;
    let z_sqr = picked_normal.z * picked_normal.z;

    let added = y_sqr + (x_sqr / alpha_sqr.x) + (z_sqr / alpha_sqr.y);

    let denom = added * added * roughness.x * roughness.y * PI;

    return heviside / max(denom, 0.01);
}

fn ggx_Lambda(out: vec3<f32>, roughness: vec2<f32>) -> f32 {
    let alpha_sqr = roughness * roughness;

    let out_sqr = out * out;

    let inner_num = (out_sqr.x * alpha_sqr.x) + (out_sqr.z * alpha_sqr.y);
    let numerator = sqrt(1.0 + (inner_num / out_sqr.y)) - 1.0;

    return numerator / 2.0;
}

fn ggx_shadowing(in: vec3<f32>, roughness: vec2<f32>) -> f32 {
    return 1.0 / (1.0 + ggx_Lambda(in, roughness));
}

fn ggx_shadowing_masking(in: vec3<f32>, out: vec3<f32>, roughness: vec2<f32>) -> f32 {
    return 1.0 / (1.0 + ggx_Lambda(in, roughness) + ggx_Lambda(out, roughness));
}

fn ggx_visible_D(in: vec3<f32>, picked_normal: vec3<f32>, roughness: vec2<f32>) -> f32 {
    return (ggx_shadowing(in, roughness) * max(0.0, dot(in, picked_normal))*ggx_D(picked_normal, roughness)) / in.y;
}

fn oren_nayer_improved(in: vec3<f32>, seed:u32, roughness:f32, normal:vec3<f32>, tangent:vec3<f32>, color:vec3<f32>) -> DiffuseReturn {
    let incoming = -in;
    let reflected = onb_cosine_weighted_hemisphere(seed);
    let brightness = cosine_weighted_to_oren_nayer_improved(incoming, reflected, roughness, color);

    return DiffuseReturn(brightness, from_onb(reflected, normal, tangent), reflected.y);
}

const U16_MAX = 0xFFFFu;