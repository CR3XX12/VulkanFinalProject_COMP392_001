#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 skyColor;
} ubo;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    // No output variables needed for vertex shader (color is uniform)
}
