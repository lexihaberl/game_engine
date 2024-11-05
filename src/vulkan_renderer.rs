use crate::vulkan_rs::debug;
use crate::vulkan_rs::window;
use crate::vulkan_rs::Instance;
use crate::vulkan_rs::Version;

pub struct VulkanRenderer {
    instance: Instance,
}

impl VulkanRenderer {
    pub fn new() -> VulkanRenderer {
        let required_layers = debug::get_required_layers();
        let required_extensions = window::get_required_instance_extensions();
        let instance = Instance::new(
            "Vulkan Renderer",
            Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            "Vulkan Engine",
            Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            &required_layers,
            &required_extensions,
        );
        VulkanRenderer { instance }
    }
}

impl Default for VulkanRenderer {
    fn default() -> Self {
        Self::new()
    }
}
