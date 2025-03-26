#include "sky.hpp"
#include "camera.hpp"
#include <stdexcept>
#include <cstring>

struct SkyUBO {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 skyColor;
};

// Define sky cube geometry (8 vertices, 36 indices for 12 triangles)
static const std::vector<SkyVertex> SKY_VERTICES = {
    {{-1.0f, -1.0f, -1.0f}}, // 0
    {{ 1.0f, -1.0f, -1.0f}}, // 1
    {{ 1.0f,  1.0f, -1.0f}}, // 2
    {{-1.0f,  1.0f, -1.0f}}, // 3
    {{-1.0f, -1.0f,  1.0f}}, // 4
    {{ 1.0f, -1.0f,  1.0f}}, // 5
    {{ 1.0f,  1.0f,  1.0f}}, // 6
    {{-1.0f,  1.0f,  1.0f}}  // 7
};
static const std::vector<uint16_t> SKY_INDICES = {
    // Each face of the cube (two triangles) 
    0, 1, 2,  2, 3, 0,   // face 1 (back face -Z)
    4, 5, 6,  6, 7, 4,   // face 2 (front face +Z)
    0, 3, 7,  7, 4, 0,   // face 3 (left face -X)
    1, 2, 6,  6, 5, 1,   // face 4 (right face +X)
    0, 1, 5,  5, 4, 0,   // face 5 (bottom face -Y)
    3, 2, 6,  6, 7, 3    // face 6 (top face +Y)
};

Sky::Sky(vk::Device device, vk::PhysicalDevice physicalDevice, vk::CommandPool commandPool, vk::Queue graphicsQueue, uint32_t swapChainImageCount)
    : device(device), physicalDevice(physicalDevice), commandPool(commandPool), graphicsQueue(graphicsQueue) {
    createVertexBuffer();
    createIndexBuffer();
    createDescriptorSetLayout();
    createUniformBuffers(swapChainImageCount);
    createDescriptorPool(swapChainImageCount);
    createDescriptorSets(swapChainImageCount);
}

Sky::~Sky() {
    for (size_t i = 0; i < uniformBuffers.size(); ++i) {
        device.unmapMemory(uniformBuffersMemory[i]);
        device.freeMemory(uniformBuffersMemory[i]);
        device.destroyBuffer(uniformBuffers[i]);
    }
    device.destroyDescriptorPool(descriptorPool);
    device.destroyDescriptorSetLayout(descriptorSetLayout);
    device.destroyBuffer(indexBuffer);
    device.freeMemory(indexBufferMemory);
    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);
}

uint32_t Sky::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void Sky::createVertexBuffer() {
    vk::DeviceSize bufferSize = sizeof(SKY_VERTICES[0]) * SKY_VERTICES.size();
    // Staging buffer for vertices
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = bufferSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    vk::Buffer stagingBuffer = device.createBuffer(bufferInfo);
    vk::MemoryRequirements memReq = device.getBufferMemoryRequirements(stagingBuffer);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, 
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::DeviceMemory stagingBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(stagingBuffer, stagingBufferMemory, 0);
    // Copy vertex data
    void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
    std::memcpy(data, SKY_VERTICES.data(), (size_t) bufferSize);
    device.unmapMemory(stagingBufferMemory);

    // Create device-local vertex buffer
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
    vertexBuffer = device.createBuffer(bufferInfo);
    memReq = device.getBufferMemoryRequirements(vertexBuffer);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vertexBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(vertexBuffer, vertexBufferMemory, 0);

    // Copy from staging to vertex buffer
    vk::CommandBufferAllocateInfo allocInfoCmd{};
    allocInfoCmd.level = vk::CommandBufferLevel::ePrimary;
    allocInfoCmd.commandPool = commandPool;
    allocInfoCmd.commandBufferCount = 1;
    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfoCmd)[0];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer.begin(beginInfo);
    vk::BufferCopy copyRegion{0, 0, bufferSize};
    commandBuffer.copyBuffer(stagingBuffer, vertexBuffer, copyRegion);
    commandBuffer.end();
    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();
    device.freeCommandBuffers(commandPool, commandBuffer);
    device.freeMemory(stagingBufferMemory);
    device.destroyBuffer(stagingBuffer);
}

void Sky::createIndexBuffer() {
    vk::DeviceSize bufferSize = sizeof(SKY_INDICES[0]) * SKY_INDICES.size();
    // Staging buffer for indices
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = bufferSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    vk::Buffer stagingBuffer = device.createBuffer(bufferInfo);
    vk::MemoryRequirements memReq = device.getBufferMemoryRequirements(stagingBuffer);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, 
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::DeviceMemory stagingBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(stagingBuffer, stagingBufferMemory, 0);
    void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
    std::memcpy(data, SKY_INDICES.data(), (size_t) bufferSize);
    device.unmapMemory(stagingBufferMemory);

    // Device-local index buffer
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer;
    indexBuffer = device.createBuffer(bufferInfo);
    memReq = device.getBufferMemoryRequirements(indexBuffer);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    indexBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(indexBuffer, indexBufferMemory, 0);

    // Copy staging to index buffer
    vk::CommandBufferAllocateInfo allocInfoCmd{};
    allocInfoCmd.level = vk::CommandBufferLevel::ePrimary;
    allocInfoCmd.commandPool = commandPool;
    allocInfoCmd.commandBufferCount = 1;
    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfoCmd)[0];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer.begin(beginInfo);
    vk::BufferCopy copyRegion{0, 0, bufferSize};
    commandBuffer.copyBuffer(stagingBuffer, indexBuffer, copyRegion);
    commandBuffer.end();
    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();
    device.freeCommandBuffers(commandPool, commandBuffer);
    device.freeMemory(stagingBufferMemory);
    device.destroyBuffer(stagingBuffer);
}

void Sky::createDescriptorSetLayout() {
    // Single binding: uniform buffer (MVP + color)
    vk::DescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;
    descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
}

void Sky::createUniformBuffers(uint32_t swapChainImageCount) {
    vk::DeviceSize bufferSize = sizeof(SkyUBO);
    uniformBuffers.resize(swapChainImageCount);
    uniformBuffersMemory.resize(swapChainImageCount);
    uniformBuffersMapped.resize(swapChainImageCount);
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = bufferSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    for (uint32_t i = 0; i < swapChainImageCount; i++) {
        uniformBuffers[i] = device.createBuffer(bufferInfo);
        vk::MemoryRequirements memReq = device.getBufferMemoryRequirements(uniformBuffers[i]);
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, 
                                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        uniformBuffersMemory[i] = device.allocateMemory(allocInfo);
        device.bindBufferMemory(uniformBuffers[i], uniformBuffersMemory[i], 0);
        uniformBuffersMapped[i] = device.mapMemory(uniformBuffersMemory[i], 0, bufferSize);
    }
}

void Sky::createDescriptorPool(uint32_t swapChainImageCount) {
    vk::DescriptorPoolSize poolSize{};
    poolSize.type = vk::DescriptorType::eUniformBuffer;
    poolSize.descriptorCount = swapChainImageCount;
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.maxSets = swapChainImageCount;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    descriptorPool = device.createDescriptorPool(poolInfo);
}

void Sky::createDescriptorSets(uint32_t swapChainImageCount) {
    std::vector<vk::DescriptorSetLayout> layouts(swapChainImageCount, descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = swapChainImageCount;
    allocInfo.pSetLayouts = layouts.data();
    descriptorSets = device.allocateDescriptorSets(allocInfo);
    for (uint32_t i = 0; i < swapChainImageCount; ++i) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(SkyUBO);

        vk::WriteDescriptorSet descriptorWrite{};
        descriptorWrite.dstSet = descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        device.updateDescriptorSets(descriptorWrite, nullptr);
    }
}

void Sky::bind(vk::CommandBuffer commandBuffer) {
    vk::DeviceSize offsets[] = {0};
    commandBuffer.bindVertexBuffers(0, 1, &vertexBuffer, offsets);
    commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
}

void Sky::draw(vk::CommandBuffer commandBuffer) {
    commandBuffer.drawIndexed(static_cast<uint32_t>(SKY_INDICES.size()), 1, 0, 0, 0);
}

void Sky::updateUniformBuffer(uint32_t currentImage, const Camera& camera) {
    SkyUBO ubo{};
    // Place the sky cube at the camera position (so it moves with the camera)
    ubo.model = glm::translate(glm::mat4(1.0f), camera.getPosition());
    ubo.view = camera.getViewMatrix();
    ubo.proj = camera.getProjectionMatrix();
    ubo.skyColor = glm::vec4(0.5f, 0.7f, 1.0f, 1.0f);  // solid sky blue color
    std::memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}
