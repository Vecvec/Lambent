import package::bindings::{ Reservoir, WorldMarcovChainState, };
import package::importance_sampling::shared::{ unpack_confidance, pack_confidance };

@group(0) @binding(0)
var<storage, read_write> eof_gi_reservoirs: array<Reservoir>;

override CONFIDENCE_CAP = 30u;

var<immediate> offset: u32;

@compute
@workgroup_size(256, 1, 1)
fn process_gi_reservoirs(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x + offset) >= arrayLength(&eof_gi_reservoirs) {
        return;
    }
    var reservoir = eof_gi_reservoirs[id.x + offset];
    // clamp reservoir max while also recalculating the reservoir.w
    var confidance_valid = unpack_confidance(reservoir.packed_confidance_valid);
    if confidance_valid.confidance > CONFIDENCE_CAP {
        let new_confidance = CONFIDENCE_CAP;
        let confidance_mutiplier = f32(CONFIDENCE_CAP) / f32(confidance_valid.confidance);
        confidance_valid.confidance = new_confidance;
        reservoir.w *= confidance_mutiplier;
        reservoir.packed_confidance_valid = pack_confidance(confidance_valid);
        eof_gi_reservoirs[id.x + offset] = reservoir;
    }
}

@group(0) @binding(1)
var<storage, read_write> eof_world_markov_chains: array<WorldMarcovChainState>;

fn div_ceil(num: u32, denom: u32) -> u32 {
    let divided = num / denom;
    let remainder = num % denom;
    if remainder != 0 {
        return divided + 1;
    } else {
        return divided;
    }
}

override SAMPLE_MULTIPLIER = 8u;
override SAMPLE_DIVISOR = 9u;

override SAM_MUL = min(SAMPLE_MULTIPLIER, SAMPLE_DIVISOR);

fn correct_radiance(idx: u32) {
    // Lower the number of samples in the radiance cache (maintaining the same radiance)
    // Note: we don't have to worry about locking because each invocation should only modify its own chain.
    let old_num_samples = atomicLoad(&eof_world_markov_chains[idx].num_samples);

    if old_num_samples == 0u {
        return;
    }

    let new_num_samples = div_ceil(old_num_samples * SAM_MUL, SAMPLE_DIVISOR);
    let radiance_multiplier = f32(new_num_samples) / f32(old_num_samples);

    // Correct radiance to account for less samples.
    let new_luminance_x = bitcast<f32>(atomicLoad(&eof_world_markov_chains[idx].luminance[0])) * radiance_multiplier;
    atomicStore(&eof_world_markov_chains[idx].luminance[0], bitcast<u32>(new_luminance_x));
    let new_luminance_y = bitcast<f32>(atomicLoad(&eof_world_markov_chains[idx].luminance[1])) * radiance_multiplier;
    atomicStore(&eof_world_markov_chains[idx].luminance[1], bitcast<u32>(new_luminance_y));
    let new_luminance_z = bitcast<f32>(atomicLoad(&eof_world_markov_chains[idx].luminance[2])) * radiance_multiplier;
    atomicStore(&eof_world_markov_chains[idx].luminance[2], bitcast<u32>(new_luminance_z));
    atomicStore(&eof_world_markov_chains[idx].num_samples, new_num_samples);
}

@compute
@workgroup_size(256, 1, 1)
fn process_world_markov_chains(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + offset;

    if idx >= arrayLength(&eof_world_markov_chains) {
        return;
    }

    if (eof_world_markov_chains[idx].timeout) == 0 {
        // reset the radiance cache, but since the markov chain is stored each time, instead of updated, we don't need to reset it.
        atomicStore(&eof_world_markov_chains[idx].secondary_hash, 0);
        atomicStore(&eof_world_markov_chains[idx].num_samples, 0);
        atomicStore(&eof_world_markov_chains[idx].luminance[0], 0);
        atomicStore(&eof_world_markov_chains[idx].luminance[1], 0);
        atomicStore(&eof_world_markov_chains[idx].luminance[2], 0);
        atomicStore(&eof_world_markov_chains[idx].lock, 0);
        return;
    }

    // Lower the number of samples in the radiance cache (maintaining same radiance) as well as in the markov chain.
    correct_radiance(idx);

    //let new_chain_num_samples = div_ceil(atomicLoad(&eof_world_markov_chains[idx].chain.num_samples) * SAM_MUL, SAMPLE_DIVISOR);
    //atomicStore(&eof_world_markov_chains[idx].chain.num_samples, new_chain_num_samples);
    eof_world_markov_chains[idx].timeout--;
}

