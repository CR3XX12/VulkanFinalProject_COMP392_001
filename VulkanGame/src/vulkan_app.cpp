#include "vulkan_app.hpp"
#include <set>
#include <fstream>
#include <stdexcept>
#include <iostream>

// Helper function to read SPIR-V shader code from a file into a byte buffer
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

// Helper function to create a VkShaderModule from SPIR-V bytecode
static vk::ShaderModule createShaderModule(const vk::Device& device, const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    return device.createShaderModule(createInfo);
}

// Structs for queue family indices and swap chain support details
struct QueueFamilyIndices {
    int graphicsFamily = -1;
    int presentFamily = -1;
    bool isComplete() const {
        return graphicsFamily >= 0 && presentFamily >= 0;
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

// Required device extension (swapchain)
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

VulkanApp::VulkanApp() {
    initWindow();
    initVulkan();
}

VulkanApp::~VulkanApp() {
    if (device) device.waitIdle();  // ensure no GPU work is running
    // Destroy game objects first to free their Vulkan resources
    sky.reset();
    floor.reset();
    // Clean up swap chain and associated objects
    for (auto framebuffer : swapChainFramebuffers) {
        device.destroyFramebuffer(framebuffer);
    }
    device.destroyPipeline(floorPipeline);
    device.destroyPipeline(skyPipeline);
    device.destroyPipelineLayout(floorPipelineLayout);
    device.destroyPipelineLayout(skyPipelineLayout);
    device.destroyRenderPass(renderPass);
    for (auto imageView : swapChainImageViews) {
        device.destroyImageView(imageView);
    }
    device.destroySwapchainKHR(swapChain);
    // Destroy sync objects
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        device.destroySemaphore(imageAvailableSemaphores[i]);
        device.destroySemaphore(renderFinishedSemaphores[i]);
        device.destroyFence(inFlightFences[i]);
    }
    device.destroyCommandPool(commandPool);
    device.destroy();         // destroy logical device
    if (surface) instance.destroySurfaceKHR(surface);
    instance.destroy();       // destroy Vulkan instance
    glfwDestroyWindow(window);
    glfwTerminate();
}

void VulkanApp::initWindow() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW!");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);        // no OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(800, 600, "Vulkan Scene", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window!");
    }
    // Capture the mouse cursor (for first-person camera look)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void VulkanApp::initVulkan() {
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createCommandPool();
    // Create the floor and sky objects (after commandPool is available for their use)
    floor = std::make_unique<Floor>(device, physicalDevice, commandPool, graphicsQueue, static_cast<uint32_t>(swapChainImages.size()));
    sky = std::make_unique<Sky>(device, physicalDevice, commandPool, graphicsQueue, static_cast<uint32_t>(swapChainImages.size()));
    createGraphicsPipelines();
    createFramebuffers();
    createCommandBuffers();
    createSyncObjects();
}

void VulkanApp::createInstance() {
    // Application and Vulkan API info (optional but informative)
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "VulkanGame";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "NoEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    // Required extensions for GLFW (to create Vulkan surface from window)
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    // (Validation layers omitted for brevity)

    instance = vk::createInstance(createInfo);
}

void VulkanApp::createSurface() {
    // Create a Vulkan surface for the GLFW window
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(static_cast<VkInstance>(instance), window, nullptr, &rawSurface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    surface = rawSurface;
}

// Find queue family indices for graphics and presenting
QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
    QueueFamilyIndices indices;
    auto queueFamilies = device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }
        // Check if this queue family can present to our window surface
        if (device.getSurfaceSupportKHR(i, surface)) {
            indices.presentFamily = i;
        }
        if (indices.isComplete()) break;
    }
    return indices;
}

// Query swap chain support details (formats, present modes, etc.)
SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
    SwapChainSupportDetails details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);
    return details;
}

// Check if a device supports the required extensions
bool checkDeviceExtensionSupport(vk::PhysicalDevice device) {
    auto availableExtensions = device.enumerateDeviceExtensionProperties();
    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    for (const auto& ext : availableExtensions) {
        requiredExtensions.erase(ext.extensionName);
    }
    return requiredExtensions.empty();
}

void VulkanApp::pickPhysicalDevice() {
    // Find a suitable GPU that supports Vulkan and our requirements
    auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    for (const auto& dev : devices) {
        QueueFamilyIndices indices = findQueueFamilies(dev, surface);
        bool extensionsSupported = checkDeviceExtensionSupport(dev);
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapDetails = querySwapChainSupport(dev, surface);
            swapChainAdequate = !swapDetails.formats.empty() && !swapDetails.presentModes.empty();
        }
        vk::PhysicalDeviceFeatures features = dev.getFeatures();
        // We require a graphics and present queue, swap chain support, and anisotropic filtering feature
        if (indices.isComplete() && extensionsSupported && swapChainAdequate && features.samplerAnisotropy) {
            physicalDevice = dev;
            break;
        }
    }
    if (!physicalDevice) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void VulkanApp::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);

    // Specify the queues we need (graphics and present). They might be the same family.
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { 
        static_cast<uint32_t>(indices.graphicsFamily), 
        static_cast<uint32_t>(indices.presentFamily) 
    };
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueInfo{};
        queueInfo.queueFamilyIndex = queueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueInfo);
    }

    // Enable the sampler anisotropy feature (for texture filtering)
    vk::PhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    // Create the logical device
    vk::DeviceCreateInfo createInfo{};
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    // (Validation layers for device are deprecated in Vulkan 1.2, so not set here)

    device = physicalDevice.createDevice(createInfo);
    // Retrieve the queue handles
    graphicsQueue = device.getQueue(indices.graphicsFamily, 0);
    presentQueue = device.getQueue(indices.presentFamily, 0);
}

// Helper functions to choose best surface format, present mode, and extent:
vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats) {
    for (const auto& availableFormat : formats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && 
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }
    // Default to the first available format if the preferred one is not found
    return !formats.empty() ? formats[0] : vk::SurfaceFormatKHR();
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& presentModes) {
    for (const auto& availablePresentMode : presentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            // Mailbox mode offers triple buffering (avoid tearing)
            return availablePresentMode;
        }
    }
    return vk::PresentModeKHR::eFifo;  // FIFO is always supported (VSync)
}

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* window) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        // The surface has defined the size (e.g., fullscreen), use it
        return capabilities.currentExtent;
    } else {
        // Otherwise, use the window's framebuffer size
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        vk::Extent2D actualExtent{ 
            static_cast<uint32_t>(width), 
            static_cast<uint32_t>(height) 
        };
        // Clamp to allowed extent
        actualExtent.width = std::max(capabilities.minImageExtent.width, 
                                      std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, 
                                       std::min(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}

void VulkanApp::createSwapChain() {
    SwapChainSupportDetails swapDetails = querySwapChainSupport(physicalDevice, surface);
    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapDetails.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapDetails.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapDetails.capabilities, window);

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;

    // We’ll request one more image than the minimum to implement triple buffering if possible
    uint32_t imageCount = swapDetails.capabilities.minImageCount + 1;
    if (swapDetails.capabilities.maxImageCount > 0 && imageCount > swapDetails.capabilities.maxImageCount) {
        imageCount = swapDetails.capabilities.maxImageCount;
    }

    // Swap chain creation info
    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = swapChainImageFormat;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = swapChainExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    // If graphics and present queues are from different families, we allow concurrent sharing
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);
    uint32_t queueFamilyIndices[] = { 
        static_cast<uint32_t>(indices.graphicsFamily), 
        static_cast<uint32_t>(indices.presentFamily) 
    };
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapDetails.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    swapChain = device.createSwapchainKHR(createInfo);
    swapChainImages = device.getSwapchainImagesKHR(swapChain);
}

void VulkanApp::createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = swapChainImages[i];
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = swapChainImageFormat;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        swapChainImageViews[i] = device.createImageView(viewInfo);
    }
}

void VulkanApp::createRenderPass() {
    // Attachment for the swap chain color image
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    // Subpass dependency to handle layout transitions
    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = vk::AccessFlags();  // nothing to wait on
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    renderPass = device.createRenderPass(renderPassInfo);
}

void VulkanApp::createCommandPool() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.queueFamilyIndex = indices.graphicsFamily;
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    commandPool = device.createCommandPool(poolInfo);
}

void VulkanApp::createGraphicsPipelines() {
    // Load shader bytecode for floor and sky shaders
    auto floorVertCode = readFile("shaders/floor.vert.spv");
    auto floorFragCode = readFile("shaders/floor.frag.spv");
    auto skyVertCode = readFile("shaders/sky.vert.spv");
    auto skyFragCode = readFile("shaders/sky.frag.spv");
    vk::ShaderModule floorVertModule = createShaderModule(device, floorVertCode);
    vk::ShaderModule floorFragModule = createShaderModule(device, floorFragCode);
    vk::ShaderModule skyVertModule = createShaderModule(device, skyVertCode);
    vk::ShaderModule skyFragModule = createShaderModule(device, skyFragCode);

    // Create pipeline layouts (each uses its object's descriptor set layout)
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    vk::DescriptorSetLayout floorSetLayout = floor->getDescriptorSetLayout();
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &floorSetLayout;
    floorPipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
    vk::DescriptorSetLayout skySetLayout = sky->getDescriptorSetLayout();
    pipelineLayoutInfo.pSetLayouts = &skySetLayout;
    skyPipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

    // Fixed-function stage configurations common to both pipelines
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width  = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vk::Offset2D offset(0, 0); 
    vk::Rect2D scissor;
    scissor.offset = offset;
    scissor.extent = swapChainExtent;
    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.pViewports    = &viewport;
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = &scissor;
    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;
    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = 
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;
    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // **Floor Pipeline**  
    vk::PipelineShaderStageCreateInfo floorStages[2];
    floorStages[0].stage = vk::ShaderStageFlagBits::eVertex;
    floorStages[0].module = floorVertModule;
    floorStages[0].pName = "main";
    floorStages[1].stage = vk::ShaderStageFlagBits::eFragment;
    floorStages[1].module = floorFragModule;
    floorStages[1].pName = "main";

    auto bindingDesc = Vertex::getBindingDescription();
    auto attrDesc    = Vertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
    vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();

    vk::GraphicsPipelineCreateInfo floorPipelineInfo{};
    floorPipelineInfo.stageCount = 2;
    floorPipelineInfo.pStages = floorStages;
    floorPipelineInfo.pVertexInputState = &vertexInputInfo;
    floorPipelineInfo.pInputAssemblyState = &inputAssembly;
    floorPipelineInfo.pViewportState = &viewportState;
    floorPipelineInfo.pRasterizationState = &rasterizer;
    floorPipelineInfo.pMultisampleState = &multisampling;
    floorPipelineInfo.pDepthStencilState = nullptr;
    floorPipelineInfo.pColorBlendState = &colorBlending;
    floorPipelineInfo.layout = floorPipelineLayout;
    floorPipelineInfo.renderPass = renderPass;
    floorPipelineInfo.subpass = 0;
    floorPipeline = device.createGraphicsPipeline(nullptr, floorPipelineInfo).value;
    // **Sky Pipeline** (similar to floor but with no culling so we see inside the cube)
    vk::PipelineShaderStageCreateInfo skyStages[2];
    skyStages[0].stage = vk::ShaderStageFlagBits::eVertex;
    skyStages[0].module = skyVertModule;
    skyStages[0].pName = "main";
    skyStages[1].stage = vk::ShaderStageFlagBits::eFragment;
    skyStages[1].module = skyFragModule;
    skyStages[1].pName = "main";

    auto bindingDescSky = SkyVertex::getBindingDescription();
    auto attrDescSky    = SkyVertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfoSky{};
    vertexInputInfoSky.vertexBindingDescriptionCount = 1;
    vertexInputInfoSky.pVertexBindingDescriptions = &bindingDescSky;
    vertexInputInfoSky.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescSky.size());
    vertexInputInfoSky.pVertexAttributeDescriptions = attrDescSky.data();

    vk::PipelineRasterizationStateCreateInfo rasterizerSky = rasterizer;
    rasterizerSky.cullMode = vk::CullModeFlagBits::eNone;  // disable culling to render inside of cube

    vk::GraphicsPipelineCreateInfo skyPipelineInfo{};
    skyPipelineInfo.stageCount = 2;
    skyPipelineInfo.pStages = skyStages;
    skyPipelineInfo.pVertexInputState = &vertexInputInfoSky;
    skyPipelineInfo.pInputAssemblyState = &inputAssembly;
    skyPipelineInfo.pViewportState = &viewportState;
    skyPipelineInfo.pRasterizationState = &rasterizerSky;
    skyPipelineInfo.pMultisampleState = &multisampling;
    skyPipelineInfo.pDepthStencilState = nullptr;
    skyPipelineInfo.pColorBlendState = &colorBlending;
    skyPipelineInfo.layout = skyPipelineLayout;
    skyPipelineInfo.renderPass = renderPass;
    skyPipelineInfo.subpass = 0;
    skyPipeline = device.createGraphicsPipeline(nullptr, skyPipelineInfo).value;

    // Shader modules can be destroyed after pipeline creation
    device.destroyShaderModule(floorVertModule);
    device.destroyShaderModule(floorFragModule);
    device.destroyShaderModule(skyVertModule);
    device.destroyShaderModule(skyFragModule);
}

void VulkanApp::createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        vk::ImageView attachments[] = { swapChainImageViews[i] };

        vk::FramebufferCreateInfo framebufferInfo{};
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width  = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
    }
}

void VulkanApp::createCommandBuffers() {
    commandBuffers.resize(swapChainFramebuffers.size());
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    commandBuffers = device.allocateCommandBuffers(allocInfo);

    for (size_t i = 0; i < commandBuffers.size(); i++) {
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        commandBuffers[i].begin(beginInfo);

        vk::ClearValue clearColor(vk::ClearColorValue(std::array<float,4>{0.5f, 0.7f, 1.0f, 1.0f}));
        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[i];

        vk::Offset2D offset;
        offset.x = 0;
        offset.y = 0;
        
        renderPassInfo.renderArea.offset = offset;
        renderPassInfo.renderArea.extent = swapChainExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        // **Record drawing commands**:
        // Draw Sky (background cube)
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, skyPipeline);
        vk::DescriptorSet skyDescSet = sky->getDescriptorSets()[i];
        commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, skyPipelineLayout, 0, skyDescSet, nullptr);
        sky->bind(commandBuffers[i]);
        sky->draw(commandBuffers[i]);
        // Draw Floor (textured ground)
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, floorPipeline);
        vk::DescriptorSet floorDescSet = floor->getDescriptorSets()[i];
        commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, floorPipelineLayout, 0, floorDescSet, nullptr);
        floor->bind(commandBuffers[i]);
        floor->draw(commandBuffers[i]);

        commandBuffers[i].endRenderPass();
        commandBuffers[i].end();
    }
}

void VulkanApp::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChainImages.size(), vk::Fence());  // init with null handles

    vk::SemaphoreCreateInfo semInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;  // start fences signaled so on first frame we don't block

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        imageAvailableSemaphores[i] = device.createSemaphore(semInfo);
        renderFinishedSemaphores[i] = device.createSemaphore(semInfo);
        inFlightFences[i] = device.createFence(fenceInfo);
    }
}

void VulkanApp::run() {
    float cameraSpeed = 5.0f;        // movement speed in units per second
    float mouseSensitivity = 0.2f;   // look sensitivity
    double lastTime = glfwGetTime();
    double lastX = 0.0, lastY = 0.0;
    bool firstMouse = true;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // Calculate frame time (deltaTime)
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;

        // Keyboard input (WASD movement)
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) 
            camera.moveForward(cameraSpeed * deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) 
            camera.moveForward(-cameraSpeed * deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) 
            camera.moveRight(-cameraSpeed * deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) 
            camera.moveRight(cameraSpeed * deltaTime);

        // Mouse input (yaw and pitch)
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }
        double xoffset = xpos - lastX;
        double yoffset = ypos - lastY;
        lastX = xpos;
        lastY = ypos;
        camera.rotate(static_cast<float>(-xoffset * mouseSensitivity), static_cast<float>(yoffset * mouseSensitivity));

        drawFrame();
    }

    // Wait for device to finish before exiting
    device.waitIdle();
}

void VulkanApp::drawFrame() {
    // Wait for the previous frame's fence to ensure the frame is not in flight
    device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Acquire an image from the swap chain
    uint32_t imageIndex;
    vk::Result result = device.acquireNextImageKHR(swapChain, UINT64_MAX, 
                        imageAvailableSemaphores[currentFrame], nullptr, &imageIndex);
    if (result == vk::Result::eErrorOutOfDateKHR) {
        // Swap chain is out of date (e.g., window resized) – normally we'd recreate swap chain here
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    // Wait on the fence of the image to be free (if a previous frame is using this image)
    if (imagesInFlight[imageIndex]) {
        device.waitForFences(1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    // Mark this image as now in use by the current frame
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    // Update uniform buffers with the latest camera matrices for this image
    floor->updateUniformBuffer(imageIndex, camera);
    sky->updateUniformBuffer(imageIndex, camera);

    // Submit the recorded command buffer for this image to the graphics queue
    vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

    vk::SubmitInfo submitInfo{};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    device.resetFences(1, &inFlightFences[currentFrame]);
    graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

    // Present the image to the swap chain
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapChain;
    presentInfo.pImageIndices = &imageIndex;

    result = presentQueue.presentKHR(presentInfo);
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    // Advance to the next frame (for double buffering)
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}
