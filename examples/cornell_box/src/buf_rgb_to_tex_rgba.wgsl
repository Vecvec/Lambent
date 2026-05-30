@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@vertex
fn vertex(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    switch idx {
        case 0u: {
            return vec4<f32>(3.0, -1.0, 0.0, 1.0);
        }
        case 1u: {
            return vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        }
        default: {
            return vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        }
    }
}

override SIZE: u32;

@fragment
fn average(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let id = vec2<u32>(pos.xy);
    let idx = (id.x + (id.y * SIZE)) * 3u;
    let average = vec4<f32>(input[idx], input[idx + 1u], input[idx + 2u], 1.0);
    return average;
}