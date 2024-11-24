use super::instance::Instance;
use super::instance::Version;
use super::window::Surface;
use ash::vk;
use gpu_allocator::vulkan::Allocator;
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
        let physical_devices = instance.enumerate_physical_devices();

        log::info!(
            "Found {} devices with Vulkan support",
            physical_devices.len()
        );

        let mut suitable_devices: Vec<vk::PhysicalDevice> = physical_devices
            .into_iter()
            .filter(|device| {
                Self::is_device_suitable(&instance, device, surface, self.minimum_vulkan_version)
            })
            .collect();
        log::info!("Found {} suitable devices", suitable_devices.len());

        suitable_devices
            .sort_by_key(|device| Reverse(self.get_device_suitability_score(&instance, *device)));

        if suitable_devices.is_empty() {
            panic!("No suitable devices found!")
        }

        let chosen_device = suitable_devices[0];

        let device_properties = instance.get_physical_device_properties(chosen_device);
        let device_name = device_properties.device_name_as_c_str().expect(
            "Should be able to convert device name to c_str since its a string coming from a C API",
        );

        log::info!("Choosing device {:?}", device_name);

        chosen_device
    }

    fn is_device_suitable(
        instance: &Arc<Instance>,
        device: &vk::PhysicalDevice,
        surface: &Surface,
        minimum_vulkan_version: Version,
    ) -> bool {
        let device_properties = instance.get_physical_device_properties(*device);
        let min_version_vk = minimum_vulkan_version.to_api_version();

        if min_version_vk > device_properties.api_version {
            return false;
        }

        let queue_families_supported = instance.find_queue_families(device, surface).is_complete();

        //TODO: handle extensions/features/swap_chain_support better, s.t. you dont have to specify
        //stuff twice
        let required_device_extensions: [&str; 1] = ["VK_KHR_swapchain"];
        let extensions_supported =
            Self::check_device_extension_support(instance, device, &required_device_extensions);

        let mut swapchain_adequate = false;
        if extensions_supported {
            let swap_chain_support = surface.query_support_details(device);
            swapchain_adequate = !swap_chain_support.surface_formats.is_empty()
                && !swap_chain_support.present_modes.is_empty();
        }

        let features_supported = Self::check_feature_support(instance, device);

        queue_families_supported && extensions_supported && swapchain_adequate && features_supported
    }

    fn check_device_extension_support(
        instance: &Arc<Instance>,
        device: &vk::PhysicalDevice,
        required_extensions: &[&str],
    ) -> bool {
        let supported_extensions = instance.enumerate_device_extension_properties(*device);
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

    fn check_feature_support(instance: &Arc<Instance>, device: &vk::PhysicalDevice) -> bool {
        //TODO: at some point: pass required features via param -> and check whether these
        //arbitrary features are supported
        let supported_features = instance.get_supported_features(device);

        let vulkan12_features = supported_features.vulkan12_features;
        let vulkan13_features = supported_features.vulkan13_features;

        vulkan12_features.buffer_device_address == vk::TRUE
            && vulkan12_features.descriptor_indexing == vk::TRUE
            && vulkan13_features.dynamic_rendering == vk::TRUE
            && vulkan13_features.synchronization2 == vk::TRUE
    }

    fn get_device_suitability_score(
        &self,
        instance: &Arc<Instance>,
        device: vk::PhysicalDevice,
    ) -> u64 {
        let device_properties = instance.get_physical_device_properties(device);
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

#[allow(dead_code)]
pub struct DeviceFeatures<'a> {
    pub vulkan11_features: vk::PhysicalDeviceVulkan11Features<'a>,
    pub vulkan12_features: vk::PhysicalDeviceVulkan12Features<'a>,
    pub vulkan13_features: vk::PhysicalDeviceVulkan13Features<'a>,
    pub base_features: vk::PhysicalDeviceFeatures,
}

pub struct Device {
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    handle: ash::Device,
    graphics_queue: vk::Queue,
    graphics_queue_family_idx: u32,
    presentation_queue: vk::Queue,
    presentation_queue_family_idx: u32,
}

impl Device {
    pub fn new(
        instance: Arc<Instance>,
        physical_device: &vk::PhysicalDevice,
        //required_device_features: &DeviceFeatures,
        //required_extensions: &[&str],
        surface: &Surface,
    ) -> Arc<Self> {
        let queue_family_indices = instance.find_queue_families(physical_device, surface);
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

        //TODO: handle better
        let required_extensions = ["VK_KHR_swapchain"];
        let required_extensions_cstr = required_extensions
            .iter()
            .map(|ext| std::ffi::CString::new(*ext).unwrap())
            .collect::<Vec<std::ffi::CString>>();
        let required_extension_names_raw: Vec<*const c_char> = required_extensions_cstr
            .iter()
            .map(|ext| ext.as_ptr() as *const c_char)
            .collect();
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
        let required_features = vk::PhysicalDeviceFeatures2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            p_next: &mut vulkan13_feats as *mut _ as *mut std::ffi::c_void,
            features: device_features,
            ..Default::default()
        };

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
        let logical_device = instance.create_logical_device(physical_device, &device_create_info);
        let graphics_queue = unsafe { logical_device.get_device_queue(graphics_q_fam_idx, 0) };
        let presentation_queue = unsafe { logical_device.get_device_queue(present_q_fam_idx, 0) };

        Arc::new(Device {
            instance,
            physical_device: *physical_device,
            handle: logical_device,
            graphics_queue,
            graphics_queue_family_idx: graphics_q_fam_idx,
            presentation_queue,
            presentation_queue_family_idx: present_q_fam_idx,
        })
    }

    pub fn create_command_pool(&self) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: self.graphics_queue_family_idx,
            p_next: std::ptr::null(),
            ..Default::default()
        };

        unsafe {
            self.handle
                .create_command_pool(&command_pool_create_info, None)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn create_command_buffer(&self, command_pool: vk::CommandPool) -> vk::CommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            p_next: std::ptr::null(),
            ..Default::default()
        };
        unsafe {
            *self
                .handle
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("I pray that I never run out of memory")
                .first()
                .expect("We should get atleast 1 command_buffer since count is set to 1")
        }
    }

    pub fn destroy_command_pool(&self, command_pool: vk::CommandPool) {
        unsafe {
            self.handle.destroy_command_pool(command_pool, None);
        }
    }

    pub fn get_graphics_queue_idx(&self) -> u32 {
        self.graphics_queue_family_idx
    }

    pub fn get_presentation_queue_idx(&self) -> u32 {
        self.presentation_queue_family_idx
    }

    pub fn get_presentation_queue(&self) -> vk::Queue {
        self.presentation_queue
    }

    pub fn create_image(
        &self,
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        extent: vk::Extent3D,
    ) -> vk::Image {
        let image_create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: usage_flags,
            ..Default::default()
        };

        unsafe {
            self.handle
                .create_image(&image_create_info, None)
                .expect("Device hopefully not out of memory")
        }
    }

    pub fn destroy_image(&self, image: vk::Image) {
        unsafe {
            self.handle.destroy_image(image, None);
        }
    }

    pub fn get_image_memory_requirements(&self, image: vk::Image) -> vk::MemoryRequirements {
        unsafe { self.handle.get_image_memory_requirements(image) }
    }

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
    ) -> vk::ImageView {
        let image_view_create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: std::ptr::null(),
            view_type: vk::ImageViewType::TYPE_2D,
            image,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        unsafe {
            self.handle
                .create_image_view(&image_view_create_info, None)
                .expect("Device hopefully not out of memory")
        }
    }

    pub fn create_image_views(
        &self,
        format: vk::Format,
        swapchain_images: &[vk::Image],
    ) -> Vec<vk::ImageView> {
        let mut swapchain_views: Vec<vk::ImageView> = Vec::with_capacity(swapchain_images.len());
        for image in swapchain_images.iter() {
            let create_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                p_next: std::ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                ..Default::default()
            };
            let image_view = unsafe {
                self.handle
                    .create_image_view(&create_info, None)
                    .expect("Device hopefully not out of memory")
            };
            swapchain_views.push(image_view);
        }
        swapchain_views
    }

    pub fn destroy_image_view(&self, image_view: vk::ImageView) {
        unsafe {
            self.handle.destroy_image_view(image_view, None);
        }
    }

    pub fn bind_image_memory(
        &self,
        image: vk::Image,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) {
        unsafe {
            self.handle
                .bind_image_memory(image, memory, offset)
                .expect("I pray that host is never out of memory")
        }
    }

    pub fn create_swapchain_loader(&self) -> ash::khr::swapchain::Device {
        self.instance.create_swapchain_loader(&self.handle)
    }

    pub fn create_semaphore(&self) -> vk::Semaphore {
        let semaphore_create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
            ..Default::default()
        };
        unsafe {
            self.handle
                .create_semaphore(&semaphore_create_info, None)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn destroy_semaphore(&self, semaphore: vk::Semaphore) {
        unsafe {
            self.handle.destroy_semaphore(semaphore, None);
        }
    }

    pub fn create_fence(&self, flags: vk::FenceCreateFlags) -> vk::Fence {
        let fence_create_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags,
            ..Default::default()
        };
        unsafe {
            self.handle
                .create_fence(&fence_create_info, None)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn destroy_fence(&self, fence: vk::Fence) {
        unsafe {
            self.handle.destroy_fence(fence, None);
        }
    }

    pub fn wait_for_fence(&self, fence: &vk::Fence, timeout: u64) {
        self.wait_for_fences(&[*fence], true, timeout)
    }

    pub fn wait_for_fences(&self, fences: &[vk::Fence], wait_all: bool, timeout: u64) {
        unsafe {
            self.handle
                .wait_for_fences(fences, wait_all, timeout)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn reset_fence(&self, fence: &vk::Fence) {
        self.reset_fences(&[*fence])
    }

    pub fn reset_fences(&self, fences: &[vk::Fence]) {
        unsafe {
            self.handle
                .reset_fences(fences)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn begin_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
    ) {
        let begin_command_buffer_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags,
            p_inheritance_info: std::ptr::null(),
            ..Default::default()
        };

        unsafe {
            self.handle
                .begin_command_buffer(command_buffer, &begin_command_buffer_info)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn reset_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.handle
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("I pray that I never run out of memory");
        }
    }

    pub fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.handle
                .end_command_buffer(command_buffer)
                .expect("I pray that I never run out of memory");
        }
    }

    pub fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        current_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        let image_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            //TODO: all commands is not very performant -> make it more specific at some point
            // refer to https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
            src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            src_access_mask: vk::AccessFlags2::MEMORY_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            dst_access_mask: vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
            old_layout: current_layout,
            new_layout,
            image,
            subresource_range: image_subresource_range,
            ..Default::default()
        };
        let dependancy_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &image_barrier,
            ..Default::default()
        };
        unsafe {
            self.handle
                .cmd_pipeline_barrier2(command_buffer, &dependancy_info);
        }
    }

    pub fn cmd_clear_color_image(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        clear_color: &vk::ClearColorValue,
    ) {
        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        unsafe {
            self.handle.cmd_clear_color_image(
                command_buffer,
                image,
                image_layout,
                clear_color,
                &[image_subresource_range],
            );
        }
    }

    pub fn copy_image_to_image(
        &self,
        command_buffer: vk::CommandBuffer,
        src_image: vk::Image,
        dst_image: vk::Image,
        src_size: vk::Extent2D,
        dst_size: vk::Extent2D,
    ) {
        let blit_region = vk::ImageBlit2 {
            s_type: vk::StructureType::IMAGE_BLIT_2,
            p_next: std::ptr::null(),
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: src_size.width as i32,
                    y: src_size.height as i32,
                    z: 1,
                },
            ],
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: dst_size.width as i32,
                    y: dst_size.height as i32,
                    z: 1,
                },
            ],
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                mip_level: 0,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                mip_level: 0,
            },
            ..Default::default()
        };
        let blit_info = vk::BlitImageInfo2 {
            s_type: vk::StructureType::BLIT_IMAGE_INFO_2,
            p_next: std::ptr::null(),
            src_image,
            src_image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_image,
            dst_image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            filter: vk::Filter::LINEAR,
            region_count: 1,
            p_regions: &blit_region,
            ..Default::default()
        };

        unsafe {
            self.handle.cmd_blit_image2(command_buffer, &blit_info);
        }
    }

    pub fn submit_to_graphics_queue(&self, submit_info: vk::SubmitInfo2, fence: vk::Fence) {
        unsafe {
            self.handle
                .queue_submit2(self.graphics_queue, &[submit_info], fence)
                .expect("I pray that I never run out of memory");
        }
    }

    pub fn wait_idle(&self) {
        unsafe {
            self.handle
                .device_wait_idle()
                .expect("I pray that I never run out of memory");
        }
    }

    pub fn create_allocator(&self) -> Allocator {
        self.instance
            .create_allocator(self.physical_device, self.handle.clone())
    }

    pub fn create_descriptor_set_layout(
        &self,
        layout_info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> vk::DescriptorSetLayout {
        unsafe {
            self.handle
                .create_descriptor_set_layout(layout_info, None)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn destroy_descriptor_set_layout(&self, layout: vk::DescriptorSetLayout) {
        unsafe {
            self.handle.destroy_descriptor_set_layout(layout, None);
        }
    }

    pub fn create_descriptor_pool(
        &self,
        pool_info: &vk::DescriptorPoolCreateInfo,
    ) -> vk::DescriptorPool {
        unsafe {
            self.handle
                .create_descriptor_pool(pool_info, None)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn reset_descriptor_pool(&self, pool: vk::DescriptorPool) {
        unsafe {
            self.handle
                .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                .expect("This call never returns errors.");
        }
    }

    pub fn destroy_descriptor_pool(&self, pool: vk::DescriptorPool) {
        unsafe {
            self.handle.destroy_descriptor_pool(pool, None);
        }
    }

    pub fn allocate_descriptor_sets(
        &self,
        allocate_info: &vk::DescriptorSetAllocateInfo,
    ) -> Vec<vk::DescriptorSet> {
        unsafe {
            self.handle
                .allocate_descriptor_sets(allocate_info)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn update_descriptor_sets(&self, write_sets: &[vk::WriteDescriptorSet]) {
        unsafe {
            self.handle.update_descriptor_sets(write_sets, &[]);
        }
    }

    pub fn create_shader_module(
        &self,
        create_info: &vk::ShaderModuleCreateInfo,
    ) -> vk::ShaderModule {
        unsafe {
            self.handle
                .create_shader_module(create_info, None)
                .expect("I pray that I never run out of memory and that the  shader code is valid")
        }
    }

    pub fn destroy_shader_module(&self, module: vk::ShaderModule) {
        unsafe {
            self.handle.destroy_shader_module(module, None);
        }
    }

    pub fn create_pipeline_layout(
        &self,
        create_info: &vk::PipelineLayoutCreateInfo,
    ) -> vk::PipelineLayout {
        unsafe {
            self.handle
                .create_pipeline_layout(create_info, None)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn destroy_pipeline_layout(&self, layout: vk::PipelineLayout) {
        unsafe {
            self.handle.destroy_pipeline_layout(layout, None);
        }
    }

    pub fn create_compute_pipelines(
        &self,
        create_infos: &[vk::ComputePipelineCreateInfo],
    ) -> Vec<vk::Pipeline> {
        unsafe {
            self.handle
                .create_compute_pipelines(vk::PipelineCache::null(), create_infos, None)
                .expect("I pray that I never run out of memory")
        }
    }

    pub fn destroy_pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.handle.destroy_pipeline(pipeline, None);
        }
    }

    pub fn execute_compute_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        group_counts: [u32; 3],
    ) {
        unsafe {
            self.handle
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.handle.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                descriptor_sets,
                &[],
            );
            self.handle.cmd_dispatch(
                command_buffer,
                group_counts[0],
                group_counts[1],
                group_counts[2],
            )
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
