use crate::vulkan_rs::debug;
use crate::vulkan_rs::window;
use crate::vulkan_rs::Instance;
use crate::vulkan_rs::Version;
use std::sync::Arc;

pub struct VulkanRenderer {
    instance: Arc<Instance>,
    debug_messenger: Option<debug::DebugMessenger>,
}

impl VulkanRenderer {
    pub fn new() -> VulkanRenderer {
        let mut required_extensions = window::get_required_instance_extensions();
        let (required_layers, debug_messenger_create_info) = if cfg!(debug_assertions) {
            log::info!("Debug mode enabled, enabling validation layers");
            let required_debug_extensions = debug::get_required_extensions();
            required_extensions.extend(required_debug_extensions);
            (
                debug::get_required_layers(),
                Some(debug::DebugMessenger::fill_create_info()),
            )
        } else {
            log::info!("Debug mode disabled, not enabling validation layers");
            (vec![], None)
        };
        log::debug!("Required extensions: {:?}", required_extensions);
        log::debug!("Required layers: {:?}", required_layers);
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
            debug_messenger_create_info,
        );
        let debug_messenger = if cfg!(debug_assertions) {
            log::info!("Creating debug messenger");
            Some(debug::DebugMessenger::new(instance.clone()))
        } else {
            None
        };
        VulkanRenderer {
            instance,
            debug_messenger,
        }
    }
}

impl Default for VulkanRenderer {
    fn default() -> Self {
        Self::new()
    }
}
