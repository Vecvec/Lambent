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
};

import package::path_tracing::general::{
    Intersection,
    resample_mcmm,
    all_vmf_pdfs,
    Ray,
    create_ray,
    at,
    correct_tangent,
    fill_mc_screen_space_states,
    fill_mc_world_space_states,
    grid_load_radiance_estimate,
    camera_loc,
    background,
};

import package::importance_sampling::shared::{
    PI,
    TAU,
    DIFFUSE, METALLIC, TRANSPARENT,
};

@if(!vertex_return)
import package::bindings::{get_vertices,};

enable wgpu_ray_query;
enable wgpu_binding_array;
@if(vertex_return)
enable wgpu_ray_query_vertex_return;

override SAMPLES = 5u;

// Will be added by the compiler.
// const GET_POS: vec2<u32>;

var<immediate> seed_offset: u32;

@workgroup_size(64, 1, 1)
@compute
fn rt_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(subgroup_size) sub_size: u32) {
    _ = SAMPLES;

    var seed = seed_offset;

    var ray = create_ray(GET_POS);
    camera_loc = ray.origin;

    let screen_size = textureDimensions(output);
    let normalized_coords = (vec2<f32>(id.xy) / vec2<f32>(screen_size));
    let circle = normalized_coords * vec2(TAU, PI);
    let direction = vec3(sin(circle.x) * sin(circle.y), cos(circle.y), cos(circle.x) * sin(circle.y));

    let first_hit = ray_hit(&ray);
    let point_at = at(&ray, first_hit.t);

    fill_mc_screen_space_states(&seed, sub_size, GET_POS, screen_size);
    fill_mc_world_space_states(&seed, point_at, first_hit.normal, ray.direction);
    _ = resample_mcmm(&seed, point_at, first_hit.normal);

    let weight = all_vmf_pdfs(point_at, direction);

    if MODE == 0 {
        textureStore(output, id.xy, vec4<f32>(vec3(weight * 100.0, weight, weight / 100.0), 1.0));
    }

    ray.direction = direction;
    ray.origin = point_at;
    let second_hit = ray_hit(&ray);

    if MODE == 1 {
        textureStore(output, id.xy, vec4<f32>(vec3(weight, length(second_hit.emission), 0.0), 1.0));
    }

    if MODE == 2 {
        let brightness = length(select(background(&ray), second_hit.emission + grid_load_radiance_estimate(at(&ray, second_hit.t), second_hit.normal, ray.direction), second_hit.hit));
        textureStore(output, id.xy, vec4<f32>(vec3(weight, brightness, brightness / 50.0), 1.0));
    }
}

override T_MIN: f32;
override T_MAX: f32;

fn ray_hit(ray: ptr<function, Ray>) -> Intersection {
    var rq: ray_query_maybe_vertex_return;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, T_MIN, T_MAX, (*ray).origin, (*ray).direction));
    var aabb_normal: vec3<f32>;
    var aabb_tangent: vec3<f32>;
    while rayQueryProceed(&rq) {
        let intersection = rayQueryGetCandidateIntersection(&rq);
        if intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
            let material = materials[material_idx[intersection.instance_index].materials[intersection.primitive_index][0]];
            let texture_pos = tex_pos[material_idx[intersection.instance_index].materials[intersection.primitive_index][1]];
            let bary = vec3<f32>(intersection.barycentrics, ((1.0 - intersection.barycentrics.x) - intersection.barycentrics.y));
            let tex_coords_float = fma(unpack2x16float(texture_pos.tex_pos_base_colour_1), vec2<f32>(bary.z), fma(unpack2x16float(texture_pos.tex_pos_base_colour_2), vec2<f32>(bary.x), (unpack2x16float(texture_pos.tex_pos_base_colour_3) * bary.y)));
            let idx_diffuse = unpack_2xu16(material.tex_idx_base_colour_emission).x;
            let ty = unpack4xU8(material.ty_thin_walled_subsurface).x;
            if !near_zero(textureSampleLevel(textures[idx_diffuse], sam, tex_coords_float, intersection.t).w) || ty == TRANSPARENT {
                rayQueryConfirmIntersection(&rq);
            }
        } else if intersection.kind == RAY_QUERY_INTERSECTION_AABB {
            let aabb_intersection = intersect(intersection);
            if aabb_intersection.hit {
                aabb_normal = aabb_intersection.normal;
                aabb_tangent = aabb_intersection.tangent;
                let t = aabb_intersection.t;
                rayQueryGenerateIntersection(&rq, t);
            }
        }
    }
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var return_intersection = Intersection();
    if intersection.kind != RAY_QUERY_INTERSECTION_NONE {
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
        if intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
            var vertices: array<vec3<f32>, 3>;
            @if (vertex_return) {
                vertices = getCommittedHitVertexPositions(&rq);
            }
            @if (!vertex_return) {
                vertices = get_vertices(rayQueryGetCommittedIntersection(&rq));
            }
            tangent = correct_tangent(normalize(vertices[0] - vertices[1]));
            normal = normalize(cross(tangent, vertices[0] - vertices[2]));
        } else if intersection.kind == RAY_QUERY_INTERSECTION_GENERATED {
            tangent = aabb_tangent;
            normal = aabb_normal;
        }

        if dot((*ray).direction, normal) > 0.0 {
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
        if idx_roughness_coat.y != 0xFFFFu {
            let coat_colour = textureSampleLevel(textures[idx_roughness_coat.y], sam, coat_tex_coords_float, intersection.t);
            return_intersection.color_coat = vec4(coat_colour.xyz * recolour, coat_colour.w);
        } else {
            return_intersection.color_coat = vec4(1.0, 1.0, 1.0, 0.0);
        }

        if idx_subsurface_metalness.x != 0xFFFFu {
            let subsurface_colour = textureSampleLevel(textures[idx_subsurface_metalness.x], sam, sub_surface_tex_coords_float, intersection.t).xyz * recolour;
            return_intersection.color_subsurface = subsurface_colour;
        } else {
            return_intersection.color_subsurface = vec3(0.8, 0.8, 0.8);
        }
        if idx_subsurface_metalness.y != 0xFFFFu {
            let metalness = textureSampleLevel(textures[idx_subsurface_metalness.y], sam, metalness_tex_coords_float, intersection.t).x;
            return_intersection.metalness = metalness;
        } else {
            return_intersection.metalness = 0.0;
        }
        let emission_scale = bitcast<vec2<f32>>(unpack_2xu16(material.emission_scale) << vec2(16));
        if idx_diffuse_emission.y != 0xFFFFu {
            return_intersection.emission = textureSampleLevel(textures[idx_diffuse_emission.y], sam, emission_tex_coords_float, intersection.t).xyz * emission_scale.x;
        }
        return_intersection.emission += diffuse_colour * emission_scale.y;
        if idx_roughness_coat.x != 0xFFFFu {
            let attributes = textureSampleLevel(textures[idx_roughness_coat.x], sam, attributes_tex_coords_float, intersection.t);
            return_intersection.roughness = attributes.xy;
            return_intersection.roughness_coat = attributes.zw;
        }
        if idx_specular.x != 0xFFFFu {
            let specular_colour = textureSampleLevel(textures[idx_specular.x], sam, specular_tex_coords_float, intersection.t);
            return_intersection.color_specular = specular_colour;
        } else {
            return_intersection.color_specular = vec4(1.0);
        }
        if idx_specular.y != 0xFFFFu {
            let specular_rougness = textureSampleLevel(textures[idx_specular.y], sam, specular_tex_coords_float, intersection.t).xy;
            return_intersection.roughness_specular = specular_rougness;
        } else {
            return_intersection.roughness_specular = vec2(0.0);
        }
    }
    return return_intersection;
}