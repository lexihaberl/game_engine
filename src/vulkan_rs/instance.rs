use ash::vk;
use std::ffi::c_char;
use std::ffi::CString;
use std::sync::Arc;

pub struct Instance {
    pub entry: ash::Entry,
    pub handle: ash::Instance,
}

pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
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

impl Instance {
    pub fn new(
        app_name: &str,
        app_version: Version,
        engine_name: &str,
        engine_version: Version,
        api_version: Version,
        required_layers: &[CString],
        required_extensions: &[CString],
        debug_messenger_create_info: Option<vk::DebugUtilsMessengerCreateInfoEXT>,
    ) -> Arc<Instance> {
        let entry = unsafe { ash::Entry::load().expect("Vulkan Drivers should be installed.") };

        if !check_instance_layer_support(&entry, required_layers) {
            panic!("Required layers are not available!");
        }
        let app_name = CString::new(app_name).expect("String should not contain null byte");
        let engine_name = CString::new(engine_name).expect("String should not contain null byte");
        let app_version =
            vk::make_api_version(0, app_version.major, app_version.minor, app_version.patch);
        let engine_version = vk::make_api_version(
            0,
            engine_version.major,
            engine_version.minor,
            engine_version.patch,
        );
        let api_version =
            vk::make_api_version(0, api_version.major, api_version.minor, api_version.patch);
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
}

impl Drop for Instance {
    fn drop(&mut self) {
        log::debug!("Destroying instance!");
        unsafe {
            self.handle.destroy_instance(None);
        }
    }
}
