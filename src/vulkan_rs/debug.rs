use super::instance::Instance;
use ash::ext::debug_utils;
use ash::vk;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

pub fn get_required_layers() -> Vec<CString> {
    vec![CString::new("VK_LAYER_KHRONOS_validation")
        .expect("Hardcoded constant should not fail conversion")]
}

pub fn get_required_extensions() -> Vec<CString> {
    vec![CString::from(debug_utils::NAME)]
}

pub unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::debug!("[VK]{}{:?}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("[VK]{}{:?}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("[VK]{}{:?}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("[VK]{}{:?}", types, message),
        _ => log::error!("[VK][Unknown]{}{:?}", types, message),
    };

    vk::FALSE
}

pub struct DebugMessenger {
    _instance: Arc<Instance>,
    messenger: vk::DebugUtilsMessengerEXT,
    debug_utils_instance: debug_utils::Instance,
}

impl DebugMessenger {
    pub fn fill_create_info<'a>() -> vk::DebugUtilsMessengerCreateInfoEXT<'a> {
        vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            p_next: std::ptr::null(),
            flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(vulkan_debug_callback),
            p_user_data: std::ptr::null_mut(),
            ..Default::default()
        }
    }
    pub fn new(instance: Arc<Instance>) -> DebugMessenger {
        let create_info = Self::fill_create_info();
        let debug_utils_instance = instance.create_debug_utils_instance();
        let messenger = unsafe {
            debug_utils_instance
                .create_debug_utils_messenger(&create_info, None)
                .expect("Device should not be out of memory this early!")
        };
        DebugMessenger {
            _instance: instance,
            messenger,
            debug_utils_instance,
        }
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        log::debug!("Destroying debug messenger!");
        unsafe {
            self.debug_utils_instance
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}
