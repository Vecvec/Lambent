import package::bindings::{
    // aliases
    ray_query_maybe_vertex_return,
    // bindings
    output,
    output_normal,
    output_albedo,
    camera,
    acc_struct,
    materials,
    material_idx,
    tex_diffuse,
    sam,
    // functions
    unpack_2xu16,
    near_zero,
};

enable wgpu_ray_query;
@if(vertex_return)
enable wgpu_ray_vertex_return;

struct Ray {
    direction: vec3f,
    origin: vec3f,
}


var<immediate> seed_offset:u32;

@workgroup_size(64, 1, 1)
@compute
fn rt_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ray_sample = rt_sample(id.xy);
    textureStore(output, id.xy, vec4<f32>(ray_sample.color, 1.0));
    textureStore(output_normal, id.xy, vec4<f32>(ray_sample.normal, 1.0));
    textureStore(output_albedo, id.xy, vec4<f32>(ray_sample.albedo, 1.0));
}

struct SampleReturn {
    color: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
}

fn rt_sample(coord: vec2<u32>) -> SampleReturn {
    var ray = create_ray(coord);
    // add to sample when hitting a light source
    var sample: SampleReturn = SampleReturn();
    let hit_ty = ray_hit(ray);
    sample.color.x = hit_ty.x / 5.0;
    sample.color.y = hit_ty.y / 5.0;
    return sample;
}

override T_MIN:f32 = 0.001;
override T_MAX:f32 = 1000.0;

fn ray_hit(ray: Ray) -> vec2<f32> {
    var rq: ray_query_maybe_vertex_return;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, T_MIN, T_MAX, ray.origin, ray.direction));
    var aabb_normal: vec3<f32>;
    var aabb_tangent: vec3<f32>;
    while (rayQueryProceed(&rq)) {
        let intersection = rayQueryGetCandidateIntersection(&rq);
        if (intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
            let material = materials[material_idx[intersection.instance_index].materials[intersection.primitive_index]];
            let bary = vec3<f32>(intersection.barycentrics, ((1.0 - intersection.barycentrics.x) - intersection.barycentrics.y));
            let tex_coords_float = fma(unpack2x16float(material.tex_pos_1), vec2<f32>(bary.z), fma(unpack2x16float(material.tex_pos_2), vec2<f32>(bary.x), (unpack2x16float(material.tex_pos_3) * bary.y)));
            let idx_diffuse = unpack_2xu16(material.tex_idx_diffuse_emission).x;
            if (!near_zero(textureSampleLevel(tex_diffuse[idx_diffuse], sam, tex_coords_float, intersection.t).w)) {
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
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        // keep this in because otherwise this is not normal bindgroup layout
        let material = materials[material_idx[intersection.geometry_index].materials[intersection.primitive_index]];
        return unpack2x16float(material.refractive_index);
    }
    return vec2<f32>(0.0);
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
    // then we rotate these coordinates, but since the w location controls translation scale 0.0 will not translate
    let direction = (camera.view_inverse * vec4<f32>(projected_coords.xyz, 0.0)).xyz;
    // the position 0.0 gets rotated to 0.0, so only the tranlattion gets applied
    let position = (camera.view_inverse * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    return Ray(direction, position);
}