import super::shared::{ rand_u32, Sample, u32_to_normalised, safe_div, len_sqrd, unpack_confidance, p_hat, sam_from_res, merge, to_info, DIFFUSE, METALLIC, TRANSPARENT, SKY };
import package::bindings::{
    // Types
    MarcovChainState,
    Reservoir,
    // aliases
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
    tex_diffuse,
    tex_emission,
    tex_attributes,
    sam,
    info,
    // functions
    // Constants
    // Overrides
};

enable wgpu_ray_query;
@if(vertex_return)
enable wgpu_ray_vertex_return;

const F32_MAX = 3.40282347e+38;

@workgroup_size(64, 1, 1)
@compute
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) work_id: vec3<u32>) {
    let screen_size = textureDimensions(output);
    if (id.x > screen_size.x || id.y > screen_size.y) {
        return;
    }

    let idx = id.x + (id.y * screen_size.x);
    
    var own_seed = rand_u32((id.x * id.y)) + rand_u32(seed_offset + id.y);

    var Rs = gi_reservoirs[idx];
    var R = old_gi_reservoirs[idx];

    if dot(u32_to_normalised(R.sample_normal), u32_to_normalised(Rs.sample_normal)) < 0.95 {
        R = Reservoir();
    }

    if Rs.ty == SKY {
        return;
    }

    let w = p_hat(sam_from_res(R));
    R = merge(Rs, R, w, own_seed).reservoir;
    let confidance = unpack_confidance(R.packed_confidance_valid).confidance;
    R.W = safe_div(R.w, (f32(confidance) * p_hat(sam_from_res(R))));
    gi_reservoirs[idx] = R;
}

var<immediate> seed_offset:u32;

const U16_MAX = 0xFFFFu;