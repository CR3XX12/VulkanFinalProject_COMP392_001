#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <array>
#include <vector>

struct SkyVertex {
    glm::vec3 pos;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(SkyVertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 1> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 1> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(SkyVertex, pos);
        return attributeDescriptions;
    }
};

class Sky {
public:
    Sky(vk::Device device, vk::PhysicalDevice physicalDevice, vk::CommandPool commandPool, vk::Queue graphicsQueue, uint32_t swapChainImageCount);
    ~Sky();

    vk::DescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }
    std::vector<vk::DescriptorSet> getDescriptorSets() const { return descriptorSets; }

    void bind(vk::CommandBuffer commandBuffer);
    void draw(vk::CommandBuffer commandBuffer);
    void updateUniformBuffer(uint32_t currentImage, const struct Camera& camera);

private:
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    vk::CommandPool commandPool;
    vk::Queue graphicsQueue;

    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;
    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    void createVertexBuffer();
    void createIndexBuffer();
    void createDescriptorSetLayout();
    void createUniformBuffers(uint32_t swapChainImageCount);
    void createDescriptorPool(uint32_t swapChainImageCount);
    void createDescriptorSets(uint32_t swapChainImageCount);
};
