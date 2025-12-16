import package::bindings::{ Reservoir, CompressedInfo, MarcovChainState, near_zero };

const PI = 3.14159265358979323846264338327950288419716939937510;
const HALF_PI = PI / 2.0;
const TAU = 2.0 * PI;

const DIFFUSE = 0u;
const METALLIC = 1u;
const TRANSPARENT = 2u;
const SKY = 0xFFu;

fn len_sqrd(v:vec3<f32>) -> f32 {
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

fn vec3_near_zero(v:vec3<f32>) -> bool {
    return near_zero(v.x) && near_zero(v.y) && near_zero(v.z);
}

// converts from Carteasian coordinates to an orthonormal basis
fn to_onb(src:vec3<f32>, up:vec3<f32>, tangent:vec3<f32>) -> vec3<f32> {
    let s = tangent;
    let t = cross(up, s);
    let inv_s = cross(up, t);
    let inv_up = cross(t, s);
    let inv_t = cross(s, up);
    return fma(vec3<f32>(src.x), inv_s, fma(vec3<f32>(src.y), inv_up, (src.z * inv_t)));
}

struct Sample {
    visible_point: vec3<f32>,
    visible_normal: u32,
    sample_point: vec3<f32>,
    sample_normal: u32,
    out_radiance: vec3<f32>,
    ty: u32,
    roughness: f32,
    pdf:f32,
    //rand: array<u32, BOUNCES>,
}

struct WorkgroupLights {
    samples: array<MarcovChainState>
}

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

fn update(s_new:Sample, w_new:f32, reservoir: Reservoir, seed:u32, always_update:bool) -> Reservoir {
    var new_reservoir = reservoir;
    new_reservoir.w = new_reservoir.w + w_new;
    var confidance_valid = unpack_confidance(new_reservoir.packed_confidance_valid);
    confidance_valid.confidance++;
    new_reservoir.packed_confidance_valid = pack_confidance(confidance_valid);
    if (rand_f32(seed) * new_reservoir.w < w_new) {
        assign_sam_to_res(&new_reservoir, s_new);
    }
    return new_reservoir;
}

struct MergeReturn {
    reservoir: Reservoir,
    picked_r: bool,
}

fn merge(reservoir: Reservoir, r: Reservoir, p: f32, seed:u32) -> MergeReturn {
    var picked_r = false;
    var new_reservoir = reservoir;
    var confidance_valid = unpack_confidance(new_reservoir.packed_confidance_valid);
    let other_confidance_valid = unpack_confidance(r.packed_confidance_valid);
    let w_new = p * r.W * f32(other_confidance_valid.confidance);
    confidance_valid.confidance += other_confidance_valid.confidance;
    new_reservoir.packed_confidance_valid = pack_confidance(confidance_valid);
    new_reservoir.w = new_reservoir.w + w_new;
    // This is exquivelent to what is done in the paper but avoids `INF`s and `NAN`s which are undefined in WGSL.
    if (rand_f32(seed) * new_reservoir.w < w_new) {
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
    reservoir.pdf = s_new.pdf;
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
        r.pdf,
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

fn p_hat(sam:Sample) -> f32 {
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

fn pdf(normal:vec3<f32>, in:vec3<f32>, out:vec3<f32>, ty:u32, albedo: vec3<f32>, roughness: f32) -> f32 {
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

fn cosine_weighted_to_oren_nayer_improved(incoming: vec3<f32>, reflected: vec3<f32>, roughness:f32, color:vec3<f32>) -> vec3<f32> {
    let rad_roughness = roughness * HALF_PI;
    let rad_roughness_squared = rad_roughness * rad_roughness;
    let a = fma(0.17 * color, vec3<f32>(rad_roughness_squared / (rad_roughness_squared + 0.13)), vec3<f32>(1.0 - (0.5 * (rad_roughness_squared / (rad_roughness_squared + 0.33)))));
    let b = 0.45 * (rad_roughness_squared / (rad_roughness_squared + 0.09));
    let s = fma(-reflected.y, incoming.y, dot(reflected, incoming));
    var t = 1.0;
    if s > 0.0 {
        t = max(reflected.y, incoming.y);
        // this removes lots of weird dark spots from the image
        if (near_zero(t)) {
            t = 0.001;
        }
    }
    // this implementation uses a cosine wieghted hemisphere (weighted by cos theta r) so we do not need an extra one here
    return fma(vec3<f32>(b), vec3<f32>(s / t), a);
}

fn u32_to_normalised(u:u32) -> vec3<f32> {
    var v: vec3<f32>;
    let unpacked = unpack2x16snorm(u);
    v.x = unpacked.x;
    v.z = unpacked.y;
    v.y = 1.0 - (abs(v.x) + abs(v.z));
    if (v.y < 0.0) {
        let x = v.x;
        v.x = copy_sign(x, 1.0 - abs(v.z));
        v.z = copy_sign(v.z, 1.0 - abs(x));
    }
    return normalize(v);
}

fn float_sign(f:f32) -> f32 {
    if (f < 0.0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

fn copy_sign(src:f32, dst: f32) -> f32 {
    let abs_dst = abs(dst);
    if (src < 0.0) {
        return -abs_dst;
    } else {
        return abs_dst;
    }
}


// pseudo random number generator using a pcg hash
fn rand_u32(own_seed:u32) -> u32 {
    let state = own_seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return ((word >> 22u) ^ word);
}

const U32_MAX = 0xFFFFFFFFu;
const RECIP_U32_MAX = 1.0 / f32(U32_MAX);

fn rand_f32(own_seed:u32) -> f32 {
    var seed = own_seed;
    return rand_f32_ptr(&seed);
}

fn rand_f32_ptr(own_seed:ptr<function, u32>) -> f32 {
    *own_seed = rand_u32(*own_seed);
    // fract is to handle edge cases (divide not being perfect)
    return fract(f32(*own_seed) * RECIP_U32_MAX);
}

fn rand_vec3_f32(own_seed:ptr<function, u32>) -> vec3<f32> {
    return vec3(rand_f32_ptr(own_seed), rand_f32_ptr(own_seed), rand_f32_ptr(own_seed));
}

fn rand_xz_vec3_f32(own_seed:ptr<function, u32>) -> vec3<f32> {
    return vec3(rand_f32_ptr(own_seed), 0.0, rand_f32_ptr(own_seed));
}

fn safe_div(numerator: f32, denominator: f32) -> f32 {
    return div_or(numerator, denominator, 0.0);
}

fn div_or(numerator: f32, denominator: f32, or: f32) -> f32 {
    if (near_zero(denominator))  {
        return or;
    } else {
        return numerator / denominator;
    }
}

fn safe_div_vec3(numerator: vec3<f32>, denominator: f32) -> vec3<f32> {
    if (near_zero(denominator))  {
        return vec3<f32>(0.0);
    } else {
        return numerator / denominator;
    }
}
