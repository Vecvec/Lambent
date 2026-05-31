import package::bindings::{Reservoir, CompressedInfo, MarcovChainState, world_markov_chains, old_world_markov_chains, near_zero, INVALID_KEY};

const PI = 3.14159265358979323846264338327950288419716939937510;
const HALF_PI = PI / 2.0;
const TAU = 2.0 * PI;

const DIFFUSE = 0u;
const METALLIC = 1u;
const TRANSPARENT = 2u;
const SKY = 0xFFu;

fn len_sqrd(v: vec3<f32>) -> f32 {
    return fma(v.x, v.x, fma(v.y, v.y, (v.z * v.z)));
}

fn sin_phi(v: vec3<f32>) -> f32 {
    let sin_theta = sqrt(clamp(fma(-v.y, v.y, 1.0), 0.0, 1.0));
    if near_zero(sin_theta) {
        return 0.0;
    } else {
        return clamp(v.z / sin_theta, -1.0, 0.0);
    }
}

fn cos_phi(v: vec3<f32>) -> f32 {
    let sin_theta = sqrt(clamp(fma(-v.y, v.y, 1.0), 0.0, 1.0));
    if near_zero(sin_theta) {
        return 1.0;
    } else {
        return clamp(v.x / sin_theta, -1.0, 0.0);
    }
}

fn vec3_near_zero(v: vec3<f32>) -> bool {
    return near_zero(v.x) && near_zero(v.y) && near_zero(v.z);
}

// converts from Carteasian coordinates to an orthonormal basis
fn to_onb(src: vec3<f32>, up: vec3<f32>, tangent: vec3<f32>) -> vec3<f32> {
    let s = tangent;
    let t = cross(up, s);

    // let inv_up = reflect(up, vec3(0.0, 1.0, 0.0));
    // let inv_s = reflect(s, vec3(1.0, 0.0, 0.0));
    // let inv_t = reflect(t, vec3(0.0, 0.0, 1.0));

    // return fma(vec3<f32>(src.x), inv_s, fma(vec3<f32>(src.y), inv_up, (src.z * inv_t)));
    let new_mat = inverse(mat3x3(s, up, t));
    return fma(vec3<f32>(src.x), new_mat[0], fma(vec3<f32>(src.y), new_mat[1], (src.z * new_mat[2])));
}

fn inverse(mat: mat3x3<f32>) -> mat3x3<f32> {
    let det = (mat[0][0] * mat[1][1] * mat[2][2])
        + (mat[0][1] * mat[1][2] * mat[2][0])
        + (mat[0][2] * mat[1][0] * mat[2][1])
        - (mat[0][2] * mat[1][1] * mat[2][0])
        - (mat[0][0] * mat[1][2] * mat[2][1])
        - (mat[0][1] * mat[1][0] * mat[2][2]);

    return mat3x3(
        det2x2(mat[1][1], mat[1][2], mat[2][1], mat[2][2]), -det2x2(mat[0][1], mat[0][2], mat[2][1], mat[2][2]), det2x2(mat[0][1], mat[0][2], mat[1][1], mat[1][2]),
        -det2x2(mat[1][0], mat[1][2], mat[2][0], mat[2][2]), det2x2(mat[0][0], mat[0][2], mat[2][0], mat[2][2]), -det2x2(mat[0][0], mat[0][2], mat[1][0], mat[1][2]),
        det2x2(mat[1][0], mat[1][1], mat[2][0], mat[2][1]), -det2x2(mat[0][0], mat[0][1], mat[2][0], mat[2][1]), det2x2(mat[0][0], mat[0][1], mat[1][0], mat[1][1]),
    ) * (1.0 / det);
}

fn det2x2(one_one: f32, one_two: f32, two_one: f32, two_two: f32) -> f32 {
    return (one_one * two_two) - (one_two * two_one);
}

struct Sample {
    visible_point: vec3<f32>,
    visible_normal: u32,
    sample_point: vec3<f32>,
    sample_normal: u32,
    out_radiance: vec3<f32>,
    ty: u32,
    roughness: f32,
    //rand: array<u32, BOUNCES>,
}

struct WorkgroupLights {
    samples: array<MarcovChainState>}

const MAX_CONFIDANCE = 0x7FFFFFFFu;

struct ConfidanceValid {
    confidance: u32,
    valid: u32,
}

fn unpack_confidance(packed_confidance_valid: u32) -> ConfidanceValid {
    let confidance = packed_confidance_valid & MAX_CONFIDANCE;
    let valid = (packed_confidance_valid & (~MAX_CONFIDANCE)) >> 31;
    return ConfidanceValid(confidance, valid);
}

fn pack_confidance(confidance_valid: ConfidanceValid) -> u32 {
    var packed_confidance_valid = confidance_valid.confidance & MAX_CONFIDANCE;
    packed_confidance_valid |= confidance_valid.valid << 31;
    return packed_confidance_valid;
}

fn update(s_new: Sample, w_new: f32, reservoir: Reservoir, seed: u32, always_update: bool) -> Reservoir {
    var new_reservoir = reservoir;
    new_reservoir.w = new_reservoir.w + w_new;
    var confidance_valid = unpack_confidance(new_reservoir.packed_confidance_valid);
    confidance_valid.confidance++;
    new_reservoir.packed_confidance_valid = pack_confidance(confidance_valid);
    if rand_f32(seed) * new_reservoir.w < w_new {
        assign_sam_to_res(&new_reservoir, s_new);
    }
    return new_reservoir;
}

struct MergeReturn {
    reservoir: Reservoir,
    picked_r: bool,
}

fn merge(reservoir: Reservoir, r: Reservoir, p: f32, seed: u32) -> MergeReturn {
    var picked_r = false;
    var new_reservoir = reservoir;
    var confidance_valid = unpack_confidance(new_reservoir.packed_confidance_valid);
    let other_confidance_valid = unpack_confidance(r.packed_confidance_valid);
    let w_new = p * r.W * f32(other_confidance_valid.confidance);
    confidance_valid.confidance += other_confidance_valid.confidance;
    new_reservoir.packed_confidance_valid = pack_confidance(confidance_valid);
    new_reservoir.w = new_reservoir.w + w_new;
    // This is exquivelent to what is done in the paper but avoids `INF`s and `NAN`s which are undefined in WGSL.
    if rand_f32(seed) * new_reservoir.w < w_new {
        assign_sam_to_res(&new_reservoir, sam_from_res(r));
        picked_r = true;
    }
    //new_reservoir.W = new_reservoir.w / (p_hat(new_reservoir.sam, u32_to_normalised(new_reservoir.sam.sample_normal), new_reservoir.sam.sample_point));
    return MergeReturn(new_reservoir, picked_r);
}

fn assign_sam_to_res(reservoir: ptr<function, Reservoir>, s_new: Sample) {
    reservoir.visible_point = s_new.visible_point;
    reservoir.visible_normal = s_new.visible_normal;
    reservoir.sample_point = s_new.sample_point;
    reservoir.sample_normal = s_new.sample_normal;
    reservoir.out_radiance = s_new.out_radiance;
    reservoir.ty = s_new.ty;
    reservoir.roughness = s_new.roughness;
    var confidance_valid = unpack_confidance((*reservoir).packed_confidance_valid);
    confidance_valid.valid = 1u;
    reservoir.packed_confidance_valid = pack_confidance(confidance_valid);
}

fn sam_from_res(r: Reservoir) -> Sample {
    return Sample(
        r.visible_point,
        r.visible_normal,
        r.sample_point,
        r.sample_normal,
        r.out_radiance,
        r.ty,
        r.roughness,
    );
}

struct Info {
    cam_loc: vec3<f32>,
    emission: vec3<f32>,
    albedo: vec3<f32>,
}

fn to_info(compressed: CompressedInfo) -> Info {
    return Info(
        vec3<f32>(compressed.arr[0], compressed.arr[1], compressed.arr[2]),
        vec3<f32>(compressed.arr[3], compressed.arr[4], compressed.arr[5]),
        vec3<f32>(compressed.arr[6], compressed.arr[7], compressed.arr[8]),
    );
}

fn from_info(info: Info) -> CompressedInfo {
    return CompressedInfo(array<f32, 9>(info.cam_loc.x, info.cam_loc.y, info.cam_loc.z, info.emission.x, info.emission.y, info.emission.z, info.albedo.x, info.albedo.y, info.albedo.z));
}

fn p_hat(sam: Sample) -> f32 {
    //return 1.0 / length(sam.out_radiance);
    //return 1.0;
    //return dot(sam.out_radiance, vec3<f32>(0.333333));
    return max(length(sam.out_radiance), 0.00001);
    //return length(sam.out_radiance);
    //return length(sam.out_radiance) * brdf(decoded_normal, normalize(camera_location - sample_point), normalize(sam.visible_point - sample_point), sam.ty, albedo, sam.roughness);
    //return length(sam.out_radiance) * brdf(decoded_normal, normalize(sam.visible_point - sample_point), sam.ty) * dot(decoded_normal, normalize(sam.visible_point - sample_point));
}

/*fn brdf(normal:vec3<f32>, in:vec3<f32>, out:vec3<f32>, ty:u32, albedo: vec3<f32>, roughness: f32) -> f32 {
    switch (ty) {
        case METALLIC: {
            if (dot(normal, out) > 0.0) {
                return 1.0;//max((dot(out, reflect(-in, normal)) - 0.9) / (1.0 - 0.9), 0.0);
            } else {
                return 0.0;
            }
        }
        case TRANSPARENT: {
            return 1.0;
        }
        default: {
            if (dot(normal, out) > 0.0) {
                return (saturate(dot(normal, out)) * length(cosine_weighted_to_oren_nayer_improved(to_onb(in, normal), to_onb(out, normal), roughness, albedo))) / PI;
            } else {
                return 0.0;
            }
        }
    }
}*/

fn pdf(normal: vec3<f32>, in: vec3<f32>, out: vec3<f32>, ty: u32, albedo: vec3<f32>, roughness: f32) -> f32 {
    /*switch (ty) {
        case METALLIC: {
            if (dot(normal, out) > 0.0) {
                return 1.0;//max((dot(out, reflect(in, normal)) - 0.99) / (1.0 - 0.99), 0.0);
            } else {
                return 0.0;
            }
        }
        case TRANSPARENT: {
            return 1.0;
        }
        default: {
            if (dot(normal, out) > 0.0) {
                return saturate(dot(normal, out));
            } else {
                return 0.0;
            }
        }
    }*/
    return 1.0;
}

/// incoming is the direction of the ray.
fn cosine_weighted_to_oren_nayer_improved(incoming: vec3<f32>, normal: vec3<f32>, reflected: vec3<f32>, roughness: f32, color: vec3<f32>) -> vec3<f32> {
    let n_dot_l = dot(normal, -reflected);
    let n_dot_v = dot(normal, -incoming);
    let v_dot_l = dot(-incoming, -reflected);

    let rad_roughness = roughness;
    let rad_roughness_squared = rad_roughness * rad_roughness;
    let a = fma(0.17 * color, vec3<f32>(rad_roughness_squared / (rad_roughness_squared + 0.13)), vec3<f32>(1.0 - (0.5 * (rad_roughness_squared / (rad_roughness_squared + 0.33)))));
    let b = 0.45 * (rad_roughness_squared / (rad_roughness_squared + 0.09));
    let s = fma(n_dot_l, n_dot_v, v_dot_l);
    var t = 1.0;
    if s > 0.0 {
        t = max(n_dot_l, n_dot_v);
        // this removes lots of weird dark spots from the image
        if near_zero(t) {
            t = 0.001;
        }
    }
    // this implementation uses a cosine wieghted hemisphere (weighted by cos theta r) so we do not need an extra one here
    return fma(vec3<f32>(b), vec3<f32>(s / t), a);
}

fn u32_to_normalised(u: u32) -> vec3<f32> {
    var v: vec3<f32>;
    let unpacked = unpack2x16snorm(u);
    v.x = unpacked.x;
    v.z = unpacked.y;
    v.y = 1.0 - (abs(v.x) + abs(v.z));
    if v.y < 0.0 {
        let x = v.x;
        v.x = copy_sign(x, 1.0 - abs(v.z));
        v.z = copy_sign(v.z, 1.0 - abs(x));
    }
    return normalize(v);
}

fn float_sign(f: f32) -> f32 {
    if f < 0.0 {
        return -1.0;
    } else {
        return 1.0;
    }
}

fn copy_sign(src: f32, dst: f32) -> f32 {
    let abs_dst = abs(dst);
    if src < 0.0 {
        return -abs_dst;
    } else {
        return abs_dst;
    }
}

// pseudo random number generator using a pcg hash
fn rand_u32(own_seed: u32) -> u32 {
    let state = own_seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return ((word >> 22u) ^ word);
}

const U32_MAX = 0xFFFFFFFFu;
const RECIP_U32_MAX = 1.0 / f32(U32_MAX);

fn rand_f32(own_seed: u32) -> f32 {
    var seed = own_seed;
    return rand_f32_ptr(&seed);
}

fn rand_f32_ptr(own_seed: ptr<function, u32>) -> f32 {
    *own_seed = rand_u32(*own_seed);
    // fract is to handle edge cases (divide not being perfect)
    return fract(f32(*own_seed) * RECIP_U32_MAX);
}

fn rand_vec3_f32(own_seed: ptr<function, u32>) -> vec3<f32> {
    return vec3(rand_f32_ptr(own_seed), rand_f32_ptr(own_seed), rand_f32_ptr(own_seed));
}

fn rand_xz_vec3_f32(own_seed: ptr<function, u32>) -> vec3<f32> {
    return vec3(rand_f32_ptr(own_seed), 0.0, rand_f32_ptr(own_seed));
}

fn safe_div(numerator: f32, denominator: f32) -> f32 {
    return div_or(numerator, denominator, 0.0);
}

fn div_or(numerator: f32, denominator: f32, or: f32) -> f32 {
    if near_zero(denominator) {
        return or;
    } else {
        return numerator / denominator;
    }
}

fn safe_div_vec3(numerator: vec3<f32>, denominator: f32) -> vec3<f32> {
    if near_zero(denominator) {
        return vec3<f32>(0.0);
    } else {
        return numerator / denominator;
    }
}

// Moved the number in the paper outside to a constant.
const MARCOV_CHAIN_SAMPLE_CLAMP = 1024;

fn update_marcov_state(/* x0: vec3<f32>, */x1: vec3<f32>, x2: vec3<f32>, p_hat: f32, seed: u32, not_vmf: bool, old_state: MarcovChainState, confidence: f32, sum_mc: f32, num_resamples: u32) -> MarcovChainState {
    var state = old_state;
    if not_vmf {
        // While the origional paper used this, the newer one instead resets if it couldn't find the light
        state = MarcovChainState();
    }
    let score = p_hat;
    let rand = rand_f32(seed);
    /*if near_zero(sum_mc) {
        tentative_marcov_chain_state.num_samples = min(tentative_marcov_chain_state.num_samples + 1, MARCOV_CHAIN_SAMPLE_CLAMP);
        let alpha = max(1.0 / f32(tentative_marcov_chain_state.num_samples), 0.1);
        tentative_marcov_chain_state.weight_sum = score;
        tentative_marcov_chain_state.light_source = score * x2;
        tentative_marcov_chain_state.mean_cosine = score;
    } else */if (rand * sum_mc) < (p_hat * f32(num_resamples)) {
        if near_zero(length(get_light_dir(state, x1))) {
            return state;
        }
        state.num_samples = min(state.num_samples + 1, MARCOV_CHAIN_SAMPLE_CLAMP);
        let alpha = select(max(1.0 / f32(state.num_samples), 0.01) * confidence, 1.0, state.num_samples == 1);
        state.weight_sum = mix(state.weight_sum, score, alpha);
        state.light_source = mix(state.light_source, score * x2, alpha);
        let mean = normalize(get_light_dir(state, x1));
        //state.light_source = mix(state.light_source, x2, alpha);
        state.mean_cosine = mix(state.mean_cosine, score * dot(normalize(x2 - x1), mean), alpha);
        //state.mean_cosine = state.weight_sum * 0.9;
    }
    return state;
}

/// Returns an unnormalized vector from the `current` to the light
fn get_light_dir(state: MarcovChainState, current: vec3<f32>) -> vec3<f32> {
    return safe_div_vec3(state.light_source, state.weight_sum) - current;
    //return state.light_source - current;
}

const F32_EXP_FIELD: u32 = 0x7F800000;
const F32_SIGN_FEILD: u32 = 0x80000000;

fn strip_NaN(float: f32) -> f32 {
    let is_nan_or_inf = (bitcast<u32>(float) & F32_EXP_FIELD) == F32_EXP_FIELD;
    if is_nan_or_inf {
        let mantissa = bitcast<u32>(float) & ~(F32_EXP_FIELD | F32_SIGN_FEILD);
        if mantissa == 0 {
            return 10000.0;
        }
        return 0.0;
    } else {
        return float;
    }
}

fn strip_NaN_vec3(float: vec3<f32>) -> vec3<f32> {
    return vec3(strip_NaN(float.x), strip_NaN(float.y), strip_NaN(float.z));
}

const TOP_VEC2_u16_1_BITS = 0x80008000u;
const TOP_VEC2_u16_2_BITS = 0xC000C000u;
const TOP_VEC2_u16_4_BITS = 0xF000F000u;
const TOP_VEC2_u16_8_BITS = 0xFF00FF00u;

const QUANTIZE_BLOCK_SIZE = 2.0;
fn quantize_packed2x16snorm(initial: u32) -> u32 {
    return pack2x16snorm(clamp(round(unpack2x16snorm(initial) / QUANTIZE_BLOCK_SIZE) * QUANTIZE_BLOCK_SIZE, vec2(-1.0), vec2(1.0)));
}

// generate a (highly quantized) version of the incoming vector.
fn incoming_hash(vector: vec3<f32>) -> u32 {
    var hash = 0u;

    hash |= u32(vector.x > 0.0) << 0;
    hash |= u32(vector.y > 0.0) << 1;
    hash |= u32(vector.z > 0.0) << 2;

    return hash;
}

// Seems like the paper counts 1 as a prime.
const PRIMES = vec4<u32>(1, 687603047, 571899763, 894271171);

fn primary_hasher(loc: vec3<i32>, normal: vec3<f32>, incoming: vec3<f32>) -> u32 {
    let normal_u32 = normalised_to_u32(normal) & TOP_VEC2_u16_8_BITS;
    let incoming_u32 = rand_u32(incoming_hash(incoming));
    let to_hash = vec4<u32>(bitcast<vec3<u32>>(loc), (normal_u32));
    let hash = PRIMES * to_hash;
    return (((hash.x ^ hash.y) ^ hash.z) ^ hash.w) ^ (incoming_u32 * 840057787);
}

const SECONDARY_PRIMES = vec4<u32>(1699644091, 550105817, 956114303, 155096983);

fn secondary_hasher(loc: vec3<i32>, normal: vec3<f32>, incoming: vec3<f32>) -> u32 {
    let normal_u32 = normalised_to_u32(normal) & TOP_VEC2_u16_8_BITS;
    let incoming_u32 = 0u;//normalised_to_u32(quantize_vec3(incoming));
    let to_hash = vec4<u32>(bitcast<vec3<u32>>(loc), (normal_u32));
    let hash = SECONDARY_PRIMES * to_hash;
    // zero is invalid, imples not set.
    return ((rand_u32((hash.x ^ hash.y) ^ hash.z) ^ hash.w) ^ (incoming_u32 * 284005801)) | 1;
}

const GRID_SIZE: f32 = 0.1;
const SIZE_PER_GRID_CHANGE = 1.0;
fn grid_loc(pos: vec3<f32>, camera_loc: vec3<f32>) -> vec3<i32> {
    let diff = abs(pos - camera_loc);
    let len = max(diff.x, max(diff.y, diff.z));
    let num_shifts = u32(log2(max(len / SIZE_PER_GRID_CHANGE, 1.0)));
    let mask: u32 = (U32_MAX << min(num_shifts, 30));
    return vec3<i32>(pos / GRID_SIZE) & vec3(bitcast<i32>(mask));
}

const CACHE_TIMEOUT = 30u;

fn grid_load(pos: vec3<f32>, normal: vec3<f32>, incoming: vec3<f32>, camera_loc: vec3<f32>, force_old: bool) -> MarcovChainState {
    let grid_pos = grid_loc(pos, camera_loc);
    let hash = primary_hasher(grid_pos, normal, incoming);
    let idx = hash % arrayLength(&world_markov_chains);
    var s: MarcovChainState;
    s.light_source.x = bitcast<f32>(atomicLoad(&world_markov_chains[idx].chain.light_source[0]));
    s.light_source.y = bitcast<f32>(atomicLoad(&world_markov_chains[idx].chain.light_source[1]));
    s.light_source.z = bitcast<f32>(atomicLoad(&world_markov_chains[idx].chain.light_source[2]));
    s.mean_cosine = bitcast<f32>(atomicLoad(&world_markov_chains[idx].chain.mean_cosine));
    s.weight_sum = bitcast<f32>(atomicLoad(&world_markov_chains[idx].chain.weight_sum));
    s.num_samples = atomicLoad(&world_markov_chains[idx].chain.num_samples);
    s.score = bitcast<f32>(atomicLoad(&world_markov_chains[idx].chain.score));
    if atomicLoad(&world_markov_chains[idx].secondary_hash) != secondary_hasher(grid_pos, normal, incoming) || force_old {
        if atomicLoad(&old_world_markov_chains[idx].secondary_hash) == secondary_hasher(grid_pos, normal, incoming) {
            s.light_source.x = bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].chain.light_source[0]));
            s.light_source.y = bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].chain.light_source[1]));
            s.light_source.z = bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].chain.light_source[2]));
            s.mean_cosine = bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].chain.mean_cosine));
            s.weight_sum = bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].chain.weight_sum));
            s.num_samples = atomicLoad(&old_world_markov_chains[idx].chain.num_samples);
            s.score = bitcast<f32>(atomicLoad(&old_world_markov_chains[idx].chain.score));
        } else {
            s = MarcovChainState();
        }
    } else {
        world_markov_chains[idx].timeout = CACHE_TIMEOUT;
    }
    s.light_source.x = strip_NaN(s.light_source.x);
    s.light_source.y = strip_NaN(s.light_source.y);
    s.light_source.z = strip_NaN(s.light_source.z);
    s.mean_cosine = strip_NaN(s.mean_cosine);
    s.weight_sum = strip_NaN(s.weight_sum);
    s.score = strip_NaN(s.score);
    return s;
}

fn grid_store_markov_chain(pos: vec3<f32>, normal: vec3<f32>, incoming: vec3<f32>, camera_loc: vec3<f32>, state: MarcovChainState) {
    let grid_pos = grid_loc(pos, camera_loc);
    let hash = primary_hasher(grid_pos, normal, incoming);
    let idx = hash % arrayLength(&world_markov_chains);
    let secondary_hash = secondary_hasher(grid_pos, normal, incoming);
    if atomicLoad(&world_markov_chains[idx].secondary_hash) != secondary_hash {
        if !(atomicCompareExchangeWeak(&world_markov_chains[idx].secondary_hash, INVALID_KEY, secondary_hash).exchanged) {
            return;
        }
    }
    atomicStore(&world_markov_chains[idx].chain.light_source[0], bitcast<u32>(state.light_source.x));
    atomicStore(&world_markov_chains[idx].chain.light_source[1], bitcast<u32>(state.light_source.y));
    atomicStore(&world_markov_chains[idx].chain.light_source[2], bitcast<u32>(state.light_source.z));
    atomicStore(&world_markov_chains[idx].chain.mean_cosine, bitcast<u32>(state.mean_cosine));
    atomicStore(&world_markov_chains[idx].chain.weight_sum, bitcast<u32>(state.weight_sum));
    atomicStore(&world_markov_chains[idx].chain.num_samples, state.num_samples);
    atomicStore(&world_markov_chains[idx].chain.score, bitcast<u32>(state.score));
    world_markov_chains[idx].timeout = CACHE_TIMEOUT;
}

fn normalised_to_u32(normalized: vec3<f32>) -> u32 {
    let v = normalized / max((abs(normalized.x) + abs(normalized.y) + abs(normalized.z)), 0.01);
    var out = 0u;
    if v.y >= 0.0 {
        out = pack2x16snorm(v.xz);
    } else {
        //out = pack2x16unorm(vec2<f32>((encode_axis(v.x)), (encode_axis(v.z))));
        //let xz = (1.0 - abs(v.zx)) * vec2<f32>(float_sign(v.x), float_sign(v.z));
        let xz = vec2<f32>(copy_sign(v.x, 1.0 - abs(v.z)), copy_sign(v.z, 1.0 - abs(v.x)));
        out = pack2x16snorm(xz);
    }
    return out;
}
