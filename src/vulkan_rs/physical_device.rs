use super::instance::Instance;
use super::instance::Version;
use super::window::Surface;
use ash::vk;
use std::cmp::Reverse;
use std::sync::Arc;

pub struct PhysicalDeviceSelector {
    minimum_vulkan_version: Version,
}

impl PhysicalDeviceSelector {
    pub fn new(minimum_vulkan_version: Version) -> Self {
        PhysicalDeviceSelector {
            minimum_vulkan_version,
        }
    }

    pub fn select(&self, instance: Arc<Instance>, surface: &Surface) -> vk::PhysicalDevice {
        let physical_devices = unsafe {
            instance
                .handle
                .enumerate_physical_devices()
                .expect("We hopefully have enough memory to handle this call")
        };
        log::info!(
            "Found {} devices with Vulkan support",
            physical_devices.len()
        );

        let mut suitable_devices: Vec<vk::PhysicalDevice> = physical_devices
            .into_iter()
            .filter(|device| {
                Self::is_device_suitable(
                    &instance.handle,
                    device,
                    surface,
                    self.minimum_vulkan_version,
                )
            })
            .collect();
        log::info!("Found {} suitable devices", suitable_devices.len());

        suitable_devices.sort_by_key(|device| {
            Reverse(self.get_device_suitability_score(&instance.handle, *device))
        });

        if suitable_devices.is_empty() {
            panic!("No suitable devices found!")
        }

        let chosen_device = suitable_devices[0];

        let device_properties = unsafe {
            instance
                .handle
                .get_physical_device_properties(chosen_device)
        };
        let device_name = device_properties.device_name_as_c_str().expect(
            "Should be able to convert device name to c_str since its a string coming from a C API",
        );

        log::info!("Choosing device {:?}", device_name);

        chosen_device
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
        surface: &Surface,
        minimum_vulkan_version: Version,
    ) -> bool {
        let device_properties = unsafe { instance.get_physical_device_properties(*device) };
        let min_version_vk = minimum_vulkan_version.to_api_version();

        if min_version_vk > device_properties.api_version {
            return false;
        }

        let queue_families_supported =
            Self::find_queue_families(instance, device, surface).is_complete();

        let required_device_extensions: [&str; 1] = ["VK_KHR_swapchain"];
        let extensions_supported =
            Self::check_device_extension_support(instance, device, &required_device_extensions);

        let mut swapchain_adequate = false;
        if extensions_supported {
            let swap_chain_support =
                SwapChainSupportDetails::query_support_details(surface, device);
            swapchain_adequate = !swap_chain_support.surface_formats.is_empty()
                && !swap_chain_support.present_modes.is_empty();
        }

        let features_supported = Self::check_feature_support(instance, device);

        queue_families_supported && extensions_supported && swapchain_adequate && features_supported
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
        required_extensions: &[&str],
    ) -> bool {
        let supported_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(*device)
                .expect("Could not enumerate device extension properties")
        };
        let cross_section = supported_extensions.iter().filter(|extension_prop| {
            required_extensions.contains(
                &extension_prop
                    .extension_name_as_c_str()
                    .expect("We only use basic ASCII strings here so shouldnt fail")
                    .to_str()
                    .expect("We only use basic ASCII strings here so shouldnt fail"),
            )
        });
        cross_section.count() == required_extensions.len()
    }

    fn find_queue_families(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
        surface: &Surface,
    ) -> QueueFamilyIndices {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };
        let mut queue_family_indices = QueueFamilyIndices::new();
        for (idx, queue_family_property) in queue_family_properties.iter().enumerate() {
            if queue_family_property
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(idx as u32);
            }
            if unsafe {
                surface
                    .loader
                    .get_physical_device_surface_support(*device, idx as u32, surface.handle)
                    .expect("Host does not have enough resources or smth")
            } {
                queue_family_indices.presentation_family = Some(idx as u32);
            }
        }
        queue_family_indices
    }

    fn get_supported_features<'a>(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
    ) -> SupportedFeatures<'a> {
        let mut vulkan11_feats = vk::PhysicalDeviceVulkan11Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            ..Default::default()
        };
        let mut vulkan12_feats = vk::PhysicalDeviceVulkan12Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            p_next: &mut vulkan11_feats as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        let mut vulkan13_feats = vk::PhysicalDeviceVulkan13Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            p_next: &mut vulkan12_feats as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        let device_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };
        let mut feature2 = vk::PhysicalDeviceFeatures2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            p_next: &mut vulkan13_feats as *mut _ as *mut std::ffi::c_void,
            features: device_features,
            ..Default::default()
        };

        unsafe { instance.get_physical_device_features2(*device, &mut feature2) };
        SupportedFeatures {
            vulkan11_features: vulkan11_feats,
            vulkan12_features: vulkan12_feats,
            vulkan13_features: vulkan13_feats,
            base_features: device_features,
        }
    }

    fn check_feature_support(instance: &ash::Instance, device: &vk::PhysicalDevice) -> bool {
        //TODO: at some point: pass required features via param -> and check whether these
        //arbitrary features are supported
        let supported_features = Self::get_supported_features(instance, device);

        let vulkan12_features = supported_features.vulkan12_features;
        let vulkan13_features = supported_features.vulkan13_features;

        vulkan12_features.buffer_device_address == vk::TRUE
            && vulkan12_features.descriptor_indexing == vk::TRUE
            && vulkan13_features.dynamic_rendering == vk::TRUE
            && vulkan13_features.synchronization2 == vk::TRUE
    }

    fn get_device_suitability_score(
        &self,
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> u64 {
        let device_properties = unsafe { instance.get_physical_device_properties(device) };
        let mut score = 0;
        score += match device_properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::CPU => 10,
            _ => 0,
        };
        score
    }
}

#[derive(Debug)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    presentation_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        QueueFamilyIndices {
            graphics_family: None,
            presentation_family: None,
        }
    }
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.presentation_family.is_some()
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    surface_formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    fn query_support_details(
        surface: &Surface,
        device: &vk::PhysicalDevice,
    ) -> SwapChainSupportDetails {
        let surface_instance = &surface.loader;
        let surface = surface.handle;
        let capabilities = unsafe {
            surface_instance
                .get_physical_device_surface_capabilities(*device, surface)
                .expect("Could not get surface capabilities")
        };
        let surface_formats = unsafe {
            surface_instance
                .get_physical_device_surface_formats(*device, surface)
                .expect("Could not get surface formats")
        };
        let present_modes = unsafe {
            surface_instance
                .get_physical_device_surface_present_modes(*device, surface)
                .expect("Could not get present modes")
        };
        SwapChainSupportDetails {
            capabilities,
            surface_formats,
            present_modes,
        }
    }
}

#[allow(dead_code)]
pub struct SupportedFeatures<'a> {
    pub vulkan11_features: vk::PhysicalDeviceVulkan11Features<'a>,
    pub vulkan12_features: vk::PhysicalDeviceVulkan12Features<'a>,
    pub vulkan13_features: vk::PhysicalDeviceVulkan13Features<'a>,
    pub base_features: vk::PhysicalDeviceFeatures,
}
