use super::device::DeviceFeatures;
use super::window::Surface;
use ash::ext::debug_utils;
use ash::khr::{android_surface, wayland_surface, win32_surface, xcb_surface, xlib_surface};
use ash::vk;
use ash::vk::SurfaceKHR;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::AllocatorDebugSettings;
use raw_window_handle::RawDisplayHandle;
use raw_window_handle::RawWindowHandle;
use std::ffi::c_char;
use std::ffi::CString;
use std::sync::Arc;

pub struct Instance {
    entry: ash::Entry,
    handle: ash::Instance,
}

#[derive(Copy, Clone)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn to_api_version(self) -> u32 {
        vk::make_api_version(0, self.major, self.minor, self.patch)
    }
}

fn get_available_instance_layers(entry: &ash::Entry) -> Vec<CString> {
    let layer_properties = unsafe {
        entry
            .enumerate_instance_layer_properties()
            .expect("Device should not run out of memory this early already")
    };
    let instance_layers: Vec<CString> = layer_properties
        .iter()
        .map(|prop| {
            CString::from(
                prop.layer_name_as_c_str()
                    .expect("Hardcoded layername should be a valid C String"),
            )
        })
        .collect();

    log::debug!("Available Instance Layers: ");
    log::debug!("==================");
    for layer in instance_layers.iter() {
        log::debug!("{:?}", layer);
    }
    log::debug!("==================");

    instance_layers
}

fn check_instance_layer_support(entry: &ash::Entry, required_layers: &[CString]) -> bool {
    let available_layers = get_available_instance_layers(entry);
    for required_layer in required_layers.iter() {
        if !available_layers.contains(required_layer) {
            log::error!("Required layer not available: {:?}", required_layer);
            return false;
        }
    }
    true
}

pub struct AppInfo {
    pub name: String,
    pub version: Version,
}

pub struct EngineInfo {
    pub name: String,
    pub version: Version,
    pub vulkan_version: Version,
}

impl Instance {
    pub fn new(
        app_info: AppInfo,
        engine_info: EngineInfo,
        required_layers: &[CString],
        required_extensions: &[CString],
        debug_messenger_create_info: Option<vk::DebugUtilsMessengerCreateInfoEXT>,
    ) -> Arc<Instance> {
        let entry = unsafe { ash::Entry::load().expect("Vulkan Drivers should be installed.") };

        if !check_instance_layer_support(&entry, required_layers) {
            panic!("Required layers are not available!");
        }
        let app_name = CString::new(app_info.name).expect("String should not contain null byte");
        let engine_name =
            CString::new(engine_info.name).expect("String should not contain null byte");
        let app_version = vk::make_api_version(
            0,
            app_info.version.major,
            app_info.version.minor,
            app_info.version.patch,
        );
        let engine_version = vk::make_api_version(
            0,
            engine_info.version.major,
            engine_info.version.minor,
            engine_info.version.patch,
        );
        let api_version = vk::make_api_version(
            0,
            engine_info.vulkan_version.major,
            engine_info.vulkan_version.minor,
            engine_info.vulkan_version.patch,
        );
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_application_name: app_name.as_ptr(),
            application_version: app_version,
            p_engine_name: engine_name.as_ptr(),
            engine_version,
            api_version,
            p_next: std::ptr::null(),
            ..Default::default()
        };

        let required_extensions_raw: Vec<*const c_char> =
            required_extensions.iter().map(|ext| ext.as_ptr()).collect();
        let required_layers_raw: Vec<*const c_char> =
            required_layers.iter().map(|layer| layer.as_ptr()).collect();
        let p_next = match debug_messenger_create_info {
            Some(create_info) => {
                &create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                    as *const std::ffi::c_void
            }
            None => std::ptr::null(),
        };

        let instance_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app_info,
            enabled_extension_count: required_extensions_raw.len() as u32,
            pp_enabled_extension_names: required_extensions_raw.as_ptr(),
            p_next,
            enabled_layer_count: required_layers_raw.len() as u32,
            pp_enabled_layer_names: required_layers_raw.as_ptr(),
            ..Default::default()
        };
        log::debug!("Creating instance!");
        let instance = unsafe {
            entry
                .create_instance(&instance_info, None)
                .expect("Extensions should be supported. Layer might not be installed, but this is only relevant for devs.")
        };
        Arc::new(Instance {
            entry,
            handle: instance,
        })
    }

    pub fn enumerate_physical_devices(&self) -> Vec<vk::PhysicalDevice> {
        unsafe {
            self.handle
                .enumerate_physical_devices()
                .expect("We hopefully have enough memory to handle this call")
        }
    }

    pub fn get_physical_device_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        unsafe { self.handle.get_physical_device_properties(physical_device) }
    }

    pub fn get_physical_device_queue_family_properties(
        &self,
        physical_device: &vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.handle
                .get_physical_device_queue_family_properties(*physical_device)
        }
    }

    pub fn enumerate_device_extension_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<vk::ExtensionProperties> {
        unsafe {
            self.handle
                .enumerate_device_extension_properties(physical_device)
                .expect("We hopefully have enough memory to handle this call")
        }
    }

    pub fn get_supported_features<'a>(&self, device: &vk::PhysicalDevice) -> DeviceFeatures<'a> {
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

        unsafe {
            self.handle
                .get_physical_device_features2(*device, &mut feature2)
        };
        DeviceFeatures {
            vulkan11_features: vulkan11_feats,
            vulkan12_features: vulkan12_feats,
            vulkan13_features: vulkan13_feats,
            base_features: device_features,
        }
    }

    pub fn create_logical_device(
        &self,
        device: &vk::PhysicalDevice,
        device_create_info: &vk::DeviceCreateInfo,
    ) -> ash::Device {
        unsafe {
            self.handle
                .create_device(*device, device_create_info, None)
                .expect("Device should hopefully not be out of memory already. Features and Extensions should be supported (checked during device suitability test)!")
        }
    }

    pub fn find_queue_families(
        &self,
        device: &vk::PhysicalDevice,
        surface: &Surface,
    ) -> QueueFamilyIndices {
        let queue_family_properties = self.get_physical_device_queue_family_properties(device);
        let mut queue_family_indices = QueueFamilyIndices::new();
        for (idx, queue_family_property) in queue_family_properties.iter().enumerate() {
            if queue_family_property
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(idx as u32);
            }
            if surface.get_physical_device_surface_support(device, idx as u32) {
                queue_family_indices.presentation_family = Some(idx as u32);
            }
        }
        queue_family_indices
    }

    pub fn create_swapchain_loader(&self, device: &ash::Device) -> ash::khr::swapchain::Device {
        ash::khr::swapchain::Device::new(&self.handle, device)
    }

    pub fn create_debug_utils_instance(&self) -> debug_utils::Instance {
        debug_utils::Instance::new(&self.entry, &self.handle)
    }

    pub fn create_surface(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> SurfaceKHR {
        let surface_opt = match (display_handle, window_handle) {
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
                let surface_desc = vk::Win32SurfaceCreateInfoKHR::default()
                    .hwnd(window.hwnd.get())
                    .hinstance(
                        window
                            .hinstance
                            .expect("Win32 hinstance should be available!")
                            .get(),
                    );
                let surface_fn = win32_surface::Instance::new(&self.entry, &self.handle);
                unsafe { surface_fn.create_win32_surface(&surface_desc, allocation_callbacks) }
            }

            (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
                let surface_desc = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(display.display.as_ptr())
                    .surface(window.surface.as_ptr());
                let surface_fn = wayland_surface::Instance::new(&self.entry, &self.handle);
                unsafe { surface_fn.create_wayland_surface(&surface_desc, allocation_callbacks) }
            }

            (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
                let surface_desc = vk::XlibSurfaceCreateInfoKHR::default()
                    .dpy(
                        display
                            .display
                            .expect("Xlib display should be available!")
                            .as_ptr(),
                    )
                    .window(window.window);
                let surface_fn = xlib_surface::Instance::new(&self.entry, &self.handle);
                unsafe { surface_fn.create_xlib_surface(&surface_desc, allocation_callbacks) }
            }

            (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
                let surface_desc = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(
                        display
                            .connection
                            .expect("Xcb connection should be available!")
                            .as_ptr(),
                    )
                    .window(window.window.get());
                let surface_fn = xcb_surface::Instance::new(&self.entry, &self.handle);
                unsafe { surface_fn.create_xcb_surface(&surface_desc, allocation_callbacks) }
            }

            (RawDisplayHandle::Android(_), RawWindowHandle::AndroidNdk(window)) => {
                let surface_desc = vk::AndroidSurfaceCreateInfoKHR::default()
                    .window(window.a_native_window.as_ptr());
                let surface_fn = android_surface::Instance::new(&self.entry, &self.handle);
                unsafe { surface_fn.create_android_surface(&surface_desc, allocation_callbacks) }
            }

            // #[cfg(target_os = "macos")]
            // (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) => {
            //     use raw_window_metal::{appkit, Layer};
            //
            //     let layer = match appkit::metal_layer_from_handle(window) {
            //         Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
            //     };
            //
            //     let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);
            //     let surface_fn = metal_surface::Instance::new(entry, instance);
            //     surface_fn.create_metal_surface(&surface_desc, allocation_callbacks)
            // }
            //
            // #[cfg(target_os = "ios")]
            // (RawDisplayHandle::UiKit(_), RawWindowHandle::UiKit(window)) => {
            //     use raw_window_metal::{uikit, Layer};
            //
            //     let layer = match uikit::metal_layer_from_handle(window) {
            //         Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
            //     };
            //
            //     let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);
            //     let surface_fn = metal_surface::Instance::new(entry, instance);
            //     surface_fn.create_metal_surface(&surface_desc, allocation_callbacks)
            // }
            _ => panic!("Unsupported display handle"),
        };
        surface_opt.expect("Device should have enough memory!")
    }

    pub fn create_surface_loader(&self) -> ash::khr::surface::Instance {
        ash::khr::surface::Instance::new(&self.entry, &self.handle)
    }

    pub fn create_allocator(
        &self,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
    ) -> Allocator {
        Allocator::new(&AllocatorCreateDesc {
            instance: self.handle.clone(),
            device,
            physical_device,
            debug_settings: AllocatorDebugSettings {
                log_frees: true,
                log_allocations: true,
                log_stack_traces: false,
                log_leaks_on_shutdown: true,
                log_memory_information: true,
                store_stack_traces: false,
            },
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .expect("I dont even know what most of these errors mean. So :shrug:")
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        log::debug!("Destroying instance!");
        unsafe {
            self.handle.destroy_instance(None);
        }
    }
}

#[derive(Debug)]
pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub presentation_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        QueueFamilyIndices {
            graphics_family: None,
            presentation_family: None,
        }
    }
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.presentation_family.is_some()
    }
}
