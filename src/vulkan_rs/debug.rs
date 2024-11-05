use std::ffi::CString;

pub fn get_required_layers() -> Vec<CString> {
    vec![CString::new("VK_LAYER_KHRONOS_validation")
        .expect("Hardcoded constant should not fail conversion")]
}
