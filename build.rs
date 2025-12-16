use wesl::{CompileOptions, Feature, FileResolver, Resolver, StandardResolver, Wesl};

fn set_generic_params<R: Resolver>(compiler: &mut Wesl<R>, _validate: bool, rm_dead_code: bool) {
    compiler.set_options(CompileOptions {
        validate: false,
        // Currently, this project only partially uses wesl, so some wesl shaders don't have entry points, causing the entire module to be removed
        strip: rm_dead_code,
        ..Default::default()
    });
    compiler.set_feature("vertex_return", Feature::from(!cfg!(no_vertex_return)));
}

fn standard_compiler(validate: bool, rm_dead_code: bool) -> Wesl<StandardResolver> {
    let mut compiler = Wesl::new("src/");

    set_generic_params(&mut compiler, validate, rm_dead_code);

    compiler
}

mod mapping_resolver {
    use std::{collections::HashMap, path::PathBuf};

    use wesl::{ModulePath, Resolver};

    pub(crate) struct MappingResolver<R: Resolver> {
        pub(crate) map: HashMap<ModulePath, ModulePath>,
        pub(crate) resolver: R,
    }

    impl<R: Resolver> Resolver for MappingResolver<R> {
        fn resolve_source<'a>(
            &'a self,
            path: &ModulePath,
        ) -> Result<std::borrow::Cow<'a, str>, wesl::ResolveError> {
            let path = self.map.get(path).unwrap_or(path);
            self.resolver.resolve_source(path)
        }

        fn display_name(&self, path: &ModulePath) -> Option<String> {
            let path = self.map.get(path).unwrap_or(path);
            self.resolver.display_name(path)
        }

        fn fs_path(&self, path: &ModulePath) -> Option<PathBuf> {
            let path = self.map.get(path).unwrap_or(path);
            self.resolver.fs_path(path)
        }
    }
}

fn main() {
    // build bindings
    standard_compiler(false, false)
        .build_artifact(&"package::bindings".parse().unwrap(), "bindings");

    // build end of frame processor
    standard_compiler(true, true).build_artifact(
        &"package::data_buffer::end_of_frame".parse().unwrap(),
        "end_of_frame",
    );

    // partially build path tracing modules
    for setting in ["low", "medium", "high"] {
        let mut compiler =
            Wesl::new_barebones().set_custom_resolver(mapping_resolver::MappingResolver {
                map: [(
                    "package::default_mode".parse().unwrap(),
                    format!("package::path_tracing::{setting}").parse().unwrap(),
                )]
                .into(),
                resolver: FileResolver::new("src/"),
            });

        // Note: we only partially compile this, so we are missing an intersection handler function.
        set_generic_params(&mut compiler, false, true);
        compiler.build_artifact(
            &"package::path_tracing::general".parse().unwrap(),
            &format!("{setting}_path_tracing"),
        );
    }

    // build ReSTIR GI style processor
    // TODO: we may want to patially compile this if we do ray traced validation
    standard_compiler(true, true).build_artifact(
        &"package::importance_sampling::importance_sampler"
            .parse()
            .unwrap(),
        "importance_sampler",
    );

    // Build debugging shaders, also only partially built, so no validation
    standard_compiler(false, true)
        .build_artifact(&"package::debug::front_face".parse().unwrap(), "front_face");
    standard_compiler(false, true).build_artifact(
        &"package::debug::reflectance".parse().unwrap(),
        "reflectance",
    );
    standard_compiler(false, true)
        .build_artifact(&"package::debug::tangent".parse().unwrap(), "tangent");
}
