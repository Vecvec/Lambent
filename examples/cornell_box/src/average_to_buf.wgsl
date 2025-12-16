@group(0) @binding(0)
var input: binding_array<texture_2d<f32>>;

@group(1) @binding(0)
var<storage, read_write> output: array<f32>;

var<immediate> num:u32;

override SIZE: u32;

@workgroup_size(64, 1, 1)
@compute
fn average(@builtin(global_invocation_id) id: vec3<u32>) {
    var average = vec4<f32>();
    for (var i = 0u; i < num; i ++) {
        average = average + (textureLoad(input[i], id.xy, 0) / f32(num));
    }
    let idx = (id.x + (id.y * SIZE)) * 3u;
    output[idx] = average.x;
    output[idx + 1u] = average.y;
    output[idx + 2u] = average.z;
}