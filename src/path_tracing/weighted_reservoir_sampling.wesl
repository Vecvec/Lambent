import package::importance_sampling::shared::rand_f32_ptr;

struct WRS_vec3f {
    /// The currently selected element.
    current_elem: vec3<f32>,
    /// The given weight for the current element.
    current_weight: f32,
    /// The sum of weights up to the latest element.
    weight_sum: f32,
}

/// Adds the current element in the stream to s, returning the new reservoir.
fn add_to_wrs_vec3f(s: ptr<function, WRS_vec3f>, element: vec3<f32>, weight: f32, own_seed: ptr<function, u32>) {
    (*s).weight_sum += weight;
    if ((rand_f32_ptr(own_seed) * (*s).weight_sum) < weight) { // since s.weight_sum should be positive, multiplying by it on both sides should be fine
        (*s).current_weight = weight;
        (*s).current_elem = element;
    }
}

struct WRS_vec4f {
    /// The currently selected element.
    current_elem: vec4<f32>,
    /// The given weight for the current element.
    current_weight: f32,
    /// The sum of weights up to the latest element.
    weight_sum: f32,
}

/// Adds the current element in the stream to s, returning the new reservoir.
fn add_to_wrs_vec4f(s: ptr<function, WRS_vec4f>, element: vec4<f32>, weight: f32, own_seed: ptr<function, u32>) {
    (*s).weight_sum += weight;
    if ((rand_f32_ptr(own_seed) * (*s).weight_sum) < weight) { // since s.weight_sum should be positive, multiplying by it on both sides should be fine
        (*s).current_weight = weight;
        (*s).current_elem = element;
    }
}