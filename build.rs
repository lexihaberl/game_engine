use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let shader_dir = "shaders";
    let output_dir = "shaders";

    println!("cargo:rerun-if-changed={}", shader_dir);

    for entry in fs::read_dir(shader_dir)
        .expect("After git cloning, folder + permission should exist and be set correctly.")
    {
        let entry = entry.expect("Just abort if we have an io error");
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                match extension
                    .to_str()
                    .expect("Extension should exist and be valid utf-8 since we set the name")
                {
                    "vert" | "frag" | "comp" => {
                        let file_stem = path
                            .file_stem()
                            .expect("File should have a valid utf-8 stem since we name it")
                            .to_str()
                            .expect("File stem should be valid utf-8 since we set the name");
                        let ext_text = extension
                            .to_str()
                            .expect("Extension should be valid utf-8 since we set the name");
                        let output_file_name = format!("{}_{}.spv", file_stem, ext_text);
                        let output_path = Path::new(&output_dir).join(output_file_name);

                        println!("Compiling {:?}", path);

                        let status = Command::new("glslc")
                            .arg(&path)
                            .arg("-o")
                            .arg(&output_path)
                            .status()
                            .expect("glslc should not fail, since it should be installed + the shaders should be valid glsl");

                        if !status.success() {
                            panic!(
                                "Failed to compile shader: {:?}",
                                path.file_name()
                                    .expect("File should have a valid utf-8 name since we name it")
                            );
                        }
                    }
                    _ => (),
                }
            }
        }
    }
}
