use super::instance::Instance;
use super::instance::Version;
use super::window::Surface;
use ash::vk;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::ffi::c_char;
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

        let queue_families_supported = find_queue_families(instance, device, surface).is_complete();

        //TODO: handle extensions/features better
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

    fn get_supported_features<'a>(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
    ) -> DeviceFeatures<'a> {
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
        DeviceFeatures {
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
pub struct DeviceFeatures<'a> {
    pub vulkan11_features: vk::PhysicalDeviceVulkan11Features<'a>,
    pub vulkan12_features: vk::PhysicalDeviceVulkan12Features<'a>,
    pub vulkan13_features: vk::PhysicalDeviceVulkan13Features<'a>,
    pub base_features: vk::PhysicalDeviceFeatures,
}

pub struct Device {
    instance: Arc<Instance>,
    handle: ash::Device,
    graphics_queue: vk::Queue,
    presentation_queue: vk::Queue,
}

impl Device {
    pub fn new(
        instance: Arc<Instance>,
        physical_device: &vk::PhysicalDevice,
        //required_device_features: &DeviceFeatures,
        //required_extensions: &[&str],
        surface: &Surface,
    ) -> Self {
        let queue_family_indices = find_queue_families(&instance.handle, physical_device, surface);
        let graphics_q_fam_idx = queue_family_indices
            .graphics_family
            .expect("Q should exist since we checked for device suitabiity");
        let present_q_fam_idx = queue_family_indices
            .presentation_family
            .expect("Q should exist since we checked for device suitabiity");

        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(graphics_q_fam_idx);
        unique_queue_families.insert(present_q_fam_idx);
        log::debug!("Using Queue Families: {:?}", unique_queue_families);

        let mut queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::new();
        for queue_family_index in unique_queue_families {
            let device_queue_create_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                p_next: std::ptr::null(),
                queue_family_index,
                queue_count: 1,
                p_queue_priorities: [1.0].as_ptr(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                ..Default::default()
            };
            queue_create_infos.push(device_queue_create_info);
        }

        //TODO handle better
        let required_extensions = ["VK_KHR_swapchain"];
        let required_extensions_cstr = required_extensions
            .iter()
            .map(|ext| std::ffi::CString::new(*ext).unwrap())
            .collect::<Vec<std::ffi::CString>>();
        let required_extension_names_raw: Vec<*const c_char> = required_extensions_cstr
            .iter()
            .map(|ext| ext.as_ptr() as *const c_char)
            .collect();
        let required_features = Self::populate_required_device_features();

        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_next: &required_features as *const vk::PhysicalDeviceFeatures2
                as *const std::ffi::c_void,
            enabled_extension_count: required_extension_names_raw.len() as u32,
            pp_enabled_extension_names: required_extension_names_raw.as_ptr(),
            flags: vk::DeviceCreateFlags::empty(),
            ..Default::default()
        };
        let logical_device = unsafe {
            instance
                .handle
                .create_device(*physical_device, &device_create_info, None)
                .expect("Device should hopefully not be out of memory already. Features and Extensions should be supported (checked during device suitability test)!")
        };
        let graphics_queue = unsafe { logical_device.get_device_queue(graphics_q_fam_idx, 0) };
        let presentation_queue = unsafe { logical_device.get_device_queue(present_q_fam_idx, 0) };

        Device {
            instance,
            handle: logical_device,
            graphics_queue,
            presentation_queue,
        }
    }

    fn populate_required_device_features<'a>() -> vk::PhysicalDeviceFeatures2<'a> {
        let mut vulkan12_feats = vk::PhysicalDeviceVulkan12Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            buffer_device_address: vk::TRUE,
            descriptor_indexing: vk::TRUE,
            ..Default::default()
        };
        let mut vulkan13_feats = vk::PhysicalDeviceVulkan13Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            p_next: &mut vulkan12_feats as *mut _ as *mut std::ffi::c_void,
            dynamic_rendering: vk::TRUE,
            synchronization2: vk::TRUE,
            ..Default::default()
        };
        let device_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };
        vk::PhysicalDeviceFeatures2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            p_next: &mut vulkan13_feats as *mut _ as *mut std::ffi::c_void,
            features: device_features,
            ..Default::default()
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        log::debug!("Destroying device!");
        unsafe {
            self.handle.destroy_device(None);
        }
    }
}
