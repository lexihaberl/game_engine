use super::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    layout: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    pub fn new(device: Arc<Device>, layout: vk::DescriptorSetLayout) -> Self {
        Self { device, layout }
    }
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        log::debug!("Destroying descriptor set layout");
        self.device.destroy_descriptor_set_layout(self.layout);
    }
}

impl<'a> DescriptorLayoutBuilder<'a> {
    pub fn new() -> DescriptorLayoutBuilder<'a> {
        DescriptorLayoutBuilder {
            bindings: Vec::new(),
        }
    }

    pub fn add_binding(
        &mut self,
        binding_idx: u32,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) {
        let binding = vk::DescriptorSetLayoutBinding {
            binding: binding_idx,
            descriptor_type,
            descriptor_count: 1,
            stage_flags,
            ..Default::default()
        };
        self.bindings.push(binding);
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn build(
        &self,
        device: Arc<Device>,
        flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> DescriptorSetLayout {
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            p_bindings: self.bindings.as_ptr(),
            binding_count: self.bindings.len() as u32,
            flags,
            ..Default::default()
        };
        let set_layout = device.create_descriptor_set_layout(&layout_info);
        DescriptorSetLayout::new(device, set_layout)
    }
}

pub struct PoolSizeRatio {
    pub descriptor_type: vk::DescriptorType,
    pub ratio: f32,
}
pub struct DescriptorAllocator {
    device: Arc<Device>,
    pool: Option<vk::DescriptorPool>,
}

impl DescriptorAllocator {
    pub fn new(device: Arc<Device>) -> DescriptorAllocator {
        Self { device, pool: None }
    }

    pub fn init_pool(&mut self, max_sets: u32, pool_ratios: &[PoolSizeRatio]) {
        let mut pool_sizes = Vec::with_capacity(pool_ratios.len());
        for pool_ratio in pool_ratios {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: pool_ratio.descriptor_type,
                descriptor_count: (max_sets as f32 * pool_ratio.ratio) as u32,
            });
        }
        let pool_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            max_sets,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            p_next: std::ptr::null(),
            ..Default::default()
        };
        self.pool = Some(self.device.create_descriptor_pool(&pool_info));
    }

    pub fn clear_descriptors(&self) {
        if let Some(pool) = self.pool {
            self.device.reset_descriptor_pool(pool);
        } else {
            panic!("Tried to clear non-initialized descriptor pool");
        }
    }

    pub fn destroy_pool(&mut self) {
        if let Some(pool) = self.pool.take() {
            self.device.destroy_descriptor_pool(pool);
        } else {
            panic!("Tried to destroy non-initialized descriptor pool");
        }
    }

    //TODO: think of a solution to handle the dependency of the descriptor set to the pool (aka
    //make sure that descriptor set is invalidated when pool is reset/destroyed)
    pub fn allocate(&self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        if let Some(pool) = self.pool {
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &layout,
                ..Default::default()
            };
            // we should only have one element in array since we only allocate one descriptor set
            self.device.allocate_descriptor_sets(&alloc_info)[0]
        } else {
            panic!("Tried to allocate from non-initialized descriptor pool");
        }
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        log::debug!("Destroying DescriptorAllocator");
        self.destroy_pool();
    }
}
