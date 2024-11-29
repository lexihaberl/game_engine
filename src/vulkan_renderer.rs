use crate::vulkan_rs::debug;
use crate::vulkan_rs::window;
use crate::vulkan_rs::AllocatedImage;
use crate::vulkan_rs::Allocator;
use crate::vulkan_rs::AppInfo;
use crate::vulkan_rs::ComputePipeline;
use crate::vulkan_rs::DescriptorAllocator;
use crate::vulkan_rs::DescriptorLayoutBuilder;
use crate::vulkan_rs::DescriptorSetLayout;
use crate::vulkan_rs::Device;
use crate::vulkan_rs::EngineInfo;
use crate::vulkan_rs::GPUDrawPushConstants;
use crate::vulkan_rs::GPUMeshBuffers;
use crate::vulkan_rs::GraphicsPipeline;
use crate::vulkan_rs::GraphicsPipelineBuilder;
use crate::vulkan_rs::ImmediateCommandData;
use crate::vulkan_rs::Instance;
use crate::vulkan_rs::MeshAsset;
use crate::vulkan_rs::PhysicalDeviceSelector;
use crate::vulkan_rs::PoolSizeRatio;
use crate::vulkan_rs::ShaderModule;
use crate::vulkan_rs::Surface;
use crate::vulkan_rs::Swapchain;
use crate::vulkan_rs::Version;
use crate::vulkan_rs::Vertex;
use ash::vk;
use nalgebra_glm as glm;
use raw_window_handle::HasDisplayHandle;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use winit::window::Window;

pub struct FrameData {
    device: Arc<Device>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    result_presentable_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

impl FrameData {
    fn new(device: Arc<Device>) -> FrameData {
        let command_pool = device.create_command_pool();
        let command_buffer = device.create_command_buffer(command_pool);
        let image_available_semaphore = device.create_semaphore();
        let result_presentable_semaphore = device.create_semaphore();
        let in_flight_fence = device.create_fence(vk::FenceCreateFlags::SIGNALED);
        FrameData {
            device,
            command_pool,
            command_buffer,
            image_available_semaphore,
            result_presentable_semaphore,
            in_flight_fence,
        }
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        log::debug!("Dropping FrameData");
        self.device.destroy_command_pool(self.command_pool);
        self.device
            .destroy_semaphore(self.image_available_semaphore);
        self.device
            .destroy_semaphore(self.result_presentable_semaphore);
        self.device.destroy_fence(self.in_flight_fence);
    }
}

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    #[allow(dead_code)]
    allocator: Arc<Mutex<Allocator>>,
    #[allow(dead_code)]
    instance: Arc<Instance>,
    #[allow(dead_code)]
    debug_messenger: Option<debug::DebugMessenger>,
    #[allow(dead_code)]
    surface: Arc<Surface>,
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    swapchain: Swapchain,
    frame_data: Vec<FrameData>,
    frame_index: usize,
    draw_image: AllocatedImage,
    depth_image: AllocatedImage,
    descriptor_allocator: DescriptorAllocator,
    draw_image_descriptor: vk::DescriptorSet,
    draw_image_descriptor_layout: DescriptorSetLayout,
    gradient_pipeline: ComputePipeline,
    immediate_command_data: ImmediateCommandData,
    mesh_pipeline: GraphicsPipeline,
    test_meshes: Vec<MeshAsset>,
}

impl VulkanRenderer {
    pub fn new(window: Arc<Window>) -> VulkanRenderer {
        let raw_display_handle = window
            .display_handle()
            .expect("I hope window has a display handle")
            .as_raw();
        let mut required_extensions = window::get_required_instance_extensions(raw_display_handle);
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
        let min_vulkan_version = Version {
            major: 1,
            minor: 3,
            patch: 0,
        };
        let app_info = AppInfo {
            name: "Vulkan Renderer".to_string(),
            version: Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
        };
        let engine_info = EngineInfo {
            name: "Vulkan Engine".to_string(),
            version: Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            vulkan_version: min_vulkan_version,
        };
        let instance = Instance::new(
            app_info,
            engine_info,
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
        let surface = window::Surface::new(instance.clone(), window.clone());

        let physical_device_selector = PhysicalDeviceSelector::new(min_vulkan_version);
        let physical_device = physical_device_selector.select(instance.clone(), &surface);

        let device = Device::new(instance.clone(), &physical_device, &surface);

        let swapchain = surface.create_swapchain(
            &physical_device,
            device.clone(),
            window.inner_size().to_logical(window.scale_factor()),
        );

        let mut frame_data = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            frame_data.push(FrameData::new(device.clone()));
        }

        let allocator = Allocator::new(device.clone());

        let draw_extent = vk::Extent3D {
            width: window.inner_size().width,
            height: window.inner_size().height,
            depth: 1,
        };
        let draw_image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE;
        let draw_image = AllocatedImage::new(
            device.clone(),
            allocator.clone(),
            vk::Format::R16G16B16A16_SFLOAT,
            draw_image_usage,
            draw_extent,
            vk::ImageAspectFlags::COLOR,
        );
        let (draw_image_descriptor, draw_image_descriptor_layout, descriptor_allocator) =
            VulkanRenderer::init_descriptors(device.clone(), &draw_image);

        let depth_image_usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        let depth_image = AllocatedImage::new(
            device.clone(),
            allocator.clone(),
            vk::Format::D32_SFLOAT,
            depth_image_usage,
            draw_extent,
            vk::ImageAspectFlags::DEPTH,
        );

        let gradient_shader = ShaderModule::new(device.clone(), "shaders/gradient_color_comp.spv");
        let gradient_pipeline = ComputePipeline::new(
            device.clone(),
            &[draw_image_descriptor_layout.layout()],
            gradient_shader,
        );

        let mesh_frag_shader = ShaderModule::new(device.clone(), "shaders/triangle_frag.spv");
        let mesh_vert_shader = ShaderModule::new(device.clone(), "shaders/triangle_mesh_vert.spv");
        let push_constants = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<GPUDrawPushConstants>() as u32,
        };
        let mesh_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 0,
            p_set_layouts: std::ptr::null(),
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constants,
            ..Default::default()
        };
        let mesh_pipeline_layout = device.create_pipeline_layout(&mesh_pipeline_layout_info);
        let mesh_pipeline = GraphicsPipelineBuilder::new()
            .set_layout(mesh_pipeline_layout)
            .set_shaders(&mesh_frag_shader, &mesh_vert_shader)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
            .disable_multisampling()
            .disable_blending()
            .enable_depth_test(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL)
            .set_color_attachment_format(draw_image.format())
            .set_depth_format(depth_image.format())
            .build_pipeline(device.clone());

        let immediate_command_data = ImmediateCommandData::new(device.clone());

        let test_meshes = MeshAsset::load_gltf(
            device.clone(),
            allocator.clone(),
            &immediate_command_data,
            Path::new("./assets/basicmesh.glb"),
            true,
        )
        .unwrap();

        VulkanRenderer {
            surface,
            allocator,
            instance,
            debug_messenger,
            physical_device,
            device,
            swapchain,
            frame_data,
            frame_index: 0,
            draw_image,
            depth_image,
            descriptor_allocator,
            draw_image_descriptor_layout,
            draw_image_descriptor,
            gradient_pipeline,
            immediate_command_data,
            mesh_pipeline,
            test_meshes,
        }
    }

    fn init_descriptors(
        device: Arc<Device>,
        draw_image: &AllocatedImage,
    ) -> (vk::DescriptorSet, DescriptorSetLayout, DescriptorAllocator) {
        let ratio_sizes = vec![PoolSizeRatio {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ratio: 1.0,
        }];

        let mut descriptor_allocator = DescriptorAllocator::new(device.clone());
        descriptor_allocator.init_pool(10, &ratio_sizes);

        let mut builder = DescriptorLayoutBuilder::new();
        builder.add_binding(
            0,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
        );
        let draw_image_descriptor_layout =
            builder.build(device.clone(), vk::DescriptorSetLayoutCreateFlags::empty());

        let draw_image_descriptor =
            descriptor_allocator.allocate(draw_image_descriptor_layout.layout());

        let image_info = vk::DescriptorImageInfo {
            image_view: draw_image.image_view(),
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
        };

        let draw_image_write: vk::WriteDescriptorSet = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: std::ptr::null(),
            dst_binding: 0,
            dst_set: draw_image_descriptor,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &image_info,
            ..Default::default()
        };

        device.update_descriptor_sets(&[draw_image_write]);

        (
            draw_image_descriptor,
            draw_image_descriptor_layout,
            descriptor_allocator,
        )
    }

    fn get_current_frame(&self) -> &FrameData {
        &self.frame_data[self.frame_index % MAX_FRAMES_IN_FLIGHT]
    }

    pub fn draw(&mut self) {
        let current_frame = self.get_current_frame();
        // MAX_IN_FLIGHT_FRAMES is 2 => we wait for the frame before the previous one to finish.
        self.device
            .wait_for_fence(&current_frame.in_flight_fence, 1_000_000_000); //1E9 ns -> 1s
        self.device.reset_fence(&current_frame.in_flight_fence);

        let (presentation_image_index, presentation_image) = self
            .swapchain
            .acquire_next_image(current_frame.image_available_semaphore, 1_000_000_000);
        let presentation_extent = self.swapchain.extent();

        let command_buffer = current_frame.command_buffer;
        // commands are finished -> can reset command buffer
        self.device.reset_command_buffer(command_buffer);

        // draw into image with higher precision before presenting results -> more accurate colors
        let draw_image = self.draw_image.image();
        let draw_extent = self.draw_image.extent();
        let draw_extent = vk::Extent2D {
            width: draw_extent.width,
            height: draw_extent.height,
        };
        let draw_image_view = self.draw_image.image_view();

        // start recording commands
        self.device
            .begin_command_buffer(command_buffer, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.transition_image_layout(
            command_buffer,
            draw_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        self.draw_background(command_buffer, draw_extent);

        self.device.transition_image_layout(
            command_buffer,
            draw_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        self.device.transition_image_layout(
            command_buffer,
            self.depth_image.image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        self.mesh_pipeline.begin_drawing(
            command_buffer,
            draw_image_view,
            self.depth_image.image_view(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            draw_extent,
            None,
        );

        self.mesh_pipeline
            .draw(command_buffer, draw_extent, &self.test_meshes[2]);

        self.mesh_pipeline.end_drawing(command_buffer);

        self.device.transition_image_layout(
            command_buffer,
            draw_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        self.device.transition_image_layout(
            command_buffer,
            presentation_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        self.device.copy_image_to_image(
            command_buffer,
            draw_image,
            presentation_image,
            draw_extent,
            presentation_extent,
        );

        self.device.transition_image_layout(
            command_buffer,
            presentation_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        self.device.end_command_buffer(command_buffer);

        self.submit_to_queue(current_frame, current_frame.in_flight_fence);
        self.swapchain.present_image(
            current_frame.result_presentable_semaphore,
            presentation_image_index,
        );
        self.frame_index += 1;
    }

    pub fn draw_background(&self, command_buffer: vk::CommandBuffer, draw_extent: vk::Extent2D) {
        self.gradient_pipeline.execute_compute(
            command_buffer,
            &[self.draw_image_descriptor],
            draw_extent,
        )
    }

    pub fn cmd_clear_image(&self, command_buffer: vk::CommandBuffer, image: vk::Image) {
        let flash_color = (self.frame_index as f32 / 100.0).sin().abs();
        let clear_value = vk::ClearColorValue {
            float32: [0.0, 0.0, flash_color, 1.0],
        };
        self.device.cmd_clear_color_image(
            command_buffer,
            image,
            vk::ImageLayout::GENERAL,
            &clear_value,
        );
    }

    fn submit_to_queue(&self, current_frame: &FrameData, fence: vk::Fence) {
        // command_buffer: is the clear cmd buffer
        // when submitting -> we say that this cmd buffer should be executed
        // when the image_available_semaphore was signaled (i.e. the image is available)
        // and after the cmd buffer is executed, the result_presentable_semaphore will be signaled
        // so that we can present the image to the surface
        let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: current_frame.command_buffer,
            p_next: std::ptr::null(),
            ..Default::default()
        };
        let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: current_frame.image_available_semaphore,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            p_next: std::ptr::null(),
            device_index: 0,
            value: 1,
            ..Default::default()
        };
        let signal_semaphore_submit_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: current_frame.result_presentable_semaphore,
            stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
            p_next: std::ptr::null(),
            device_index: 0,
            value: 1,
            ..Default::default()
        };
        let submit_info = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            p_next: std::ptr::null(),
            wait_semaphore_info_count: 1,
            p_wait_semaphore_infos: &wait_semaphore_submit_info,
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: &signal_semaphore_submit_info,
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cmd_buffer_submit_info,
            ..Default::default()
        };
        self.device.submit_to_graphics_queue(submit_info, fence);
    }

    pub fn wait_idle(&self) {
        self.device.wait_idle();
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        log::debug!("Dropping VulkanRenderer. Waiting for device idle");
        self.device.wait_idle();
        log::debug!("Device is idle. Dropping resources");
    }
}
