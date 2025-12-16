override SAMPLES = 5u;

struct MarcovChainState {
    light_source: vec3<f32>,
    mean_cosine: f32,
    weight_sum: f32,
    num_samples: u32,
    score: f32,
}

struct AtomicMarcovChainState {
    light_source: array<atomic<u32>, 3>,
    mean_cosine: atomic<u32>,
    weight_sum: atomic<u32>,
    num_samples: atomic<u32>,
    score: atomic<u32>,
}

// To overflow a f32 with this many samples would require an average (not peak) brightness of 
// > 1,267,650,530,739,316,831,303,078,052,785.5 (~1 nonillion), at that point, there is probably
// a bug somewhere else that we should be worrying about.
//
// Note: at this point, f32 will be off by up to 8 when casting, due to a step size of 16. The
// error is about 0.000006%. This is probably unnoticable.
const MAX_NUM_SAMPLES: u32 = 0xFFFFFFF;
//const MAX_NUM_SAMPLES: u32 = 2;

const STATE_LOCKED: u32 = 1;
const STATE_UNLOCKED: u32 = 0;

const INVALID_KEY: u32 = 0;

struct WorldMarcovChainState {
    secondary_hash: atomic<u32>,
    num_samples: atomic<u32>,
    luminance: array<atomic<u32>, 3>,
    lock: atomic<u32>,
    chain: AtomicMarcovChainState,
    timeout: u32,
}

struct Camera {
    // this way we can have any fov easily.
    projection_inverse: mat4x4<f32>,
    view_inverse: mat4x4<f32>,
}

struct Material {
    tex_pos_1: u32,
    tex_pos_2: u32,
    tex_pos_3: u32,
    tex_pos_recolour: u32,
    tex_idx_diffuse_emission: u32,
    tex_idx_attributes_ty: u32,
    emission_scale: f32,
    refractive_index: u32,
}

struct MaterialsIdx {
    materials: array<u32>,
}

struct CompressedInfo {
    arr: array<f32, 9>,
}

@group(0) @binding(0)
var<storage> materials: array<Material>;
@group(0) @binding(1)
var<storage> material_idx: binding_array<MaterialsIdx>;

@if(vertex_return)
@group(0) @binding(2)
var acc_struct: acceleration_structure<vertex_return>;

@if(vertex_return)
alias ray_query_maybe_vertex_return = ray_query<vertex_return>;

@if(!vertex_return)
@group(0) @binding(2)
var acc_struct: acceleration_structure;

@if(!vertex_return)
struct Vertices {
    geometry_stride: vec4<u32>,
    vertices: array<vec3<f32>>,
}

@if(!vertex_return)
@group(0) @binding(3)
var<storage> vertices: binding_array<Vertices>;

@if(!vertex_return)
struct Indices {
    indices: array<u32>,
}

@if(!vertex_return)
@group(0) @binding(4)
var<storage> indices: binding_array<Indices>;

@if(!vertex_return)
fn get_vertices(intersection: RayIntersection) -> array<vec3<f32>, 3> {
    let instance_idx = intersection.instance_index;
    let vertex_index = (vertices[instance_idx].geometry_stride.x * intersection.geometry_index) + (intersection.primitive_index * 3);
    let index_1 = select(indices[instance_idx].indices[vertex_index], vertex_index, arrayLength(&indices[instance_idx].indices) == 1);
    let index_2 = select(indices[instance_idx].indices[vertex_index + 1], vertex_index + 1, arrayLength(&indices[instance_idx].indices) == 1);
    let index_3 = select(indices[instance_idx].indices[vertex_index + 2], vertex_index + 2, arrayLength(&indices[instance_idx].indices) == 1);
    return array<vec3<f32>, 3>(vertices[instance_idx].vertices[index_1], vertices[instance_idx].vertices[index_2], vertices[instance_idx].vertices[index_3]);
}

@if(!vertex_return)
alias ray_query_maybe_vertex_return = ray_query;

@group(1) @binding(0)
var<uniform> camera: Camera;
@group(1) @binding(1)
var output: texture_storage_2d<rgba32float, write>;
@group(1) @binding(2)
var output_normal: texture_storage_2d<rgba32float, write>;
@group(1) @binding(3)
var output_albedo: texture_storage_2d<rgba32float, write>;
@group(1) @binding(4)
var bg: texture_cube<f32>;
@group(1) @binding(5)
var<storage, read_write> gi_reservoirs: array<Reservoir>;
@group(1) @binding(6)
var<storage> old_gi_reservoirs: array<Reservoir>;
@group(1) @binding(7)
var<storage, read_write> info: array<CompressedInfo>;
@group(1) @binding(8)
var<storage, read_write> markov_chains: array<MarcovChainState>;
@group(1) @binding(9)
var<storage> old_markov_chains: array<MarcovChainState>;
@group(1) @binding(10)
var<storage, read_write> world_markov_chains: array<WorldMarcovChainState>;
@group(1) @binding(11)
var<storage> old_world_markov_chains: array<WorldMarcovChainState>;

@group(2) @binding(0)
var sam:sampler;
@group(2) @binding(1)
var tex_diffuse: binding_array<texture_2d<f32>>;
@group(2) @binding(2)
var tex_emission: binding_array<texture_2d<f32>>;
@group(2) @binding(3)
// r: roughness
var tex_attributes: binding_array<texture_2d<f32>>;
@group(2) @binding(4)
var tex_recolour: texture_2d<f32>;

fn unpack_2xu16(u:u32) -> vec2<u32> {
    let u1 = u & 0xFFFFu;
    let u2 = (u & 0xFFFF0000u) >> 16u;
    return vec2<u32>(u1, u2);
}

struct Reservoir {
    visible_point: vec3<f32>,   // 0 16 12
    visible_normal: u32,        // 12 4 4
    sample_point: vec3<f32>,    // 16 16 12
    sample_normal: u32,         // 28 4 4
    out_radiance: vec3<f32>,    // 32 16 12
    ty: u32,                    // 44 4 4
    roughness: f32,             // 48 4 4
    pdf:f32,                    // 52 4 4
    w: f32,                     // 56 4 4
    packed_confidance_valid: u32,    // 60 4 4
    W: f32,                     // 64 4 4
}

const SMALL_VAL:f32 = 0.0001;

fn near_zero(float:f32) -> bool {
    return (float < SMALL_VAL) && (float > -SMALL_VAL);
}