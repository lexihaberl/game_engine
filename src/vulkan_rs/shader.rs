use super::device::Device;
use ash::vk;
use std::io::Read;
use std::sync::Arc;

pub struct ShaderModule {
    device: Arc<Device>,
    module: vk::ShaderModule,
}

fn read_shader_file(path: &str) -> Vec<u8> {
    std::fs::File::open(path)
        .expect("I hope that the file exists")
        .bytes()
        .map(|byte| byte.expect("Bytecode should be valid cuz it was created by a fancy compiler"))
        .collect()
}
impl ShaderModule {
    pub fn new(device: Arc<Device>, path: &str) -> Self {
        let shader_file_bytes = read_shader_file(path);
        let create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            code_size: shader_file_bytes.len(),
            p_code: shader_file_bytes.as_ptr() as *const u32,
            ..Default::default()
        };

        let module = device.create_shader_module(&create_info);
        Self { device, module }
    }

    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        self.device.destroy_shader_module(self.module);
    }
}
