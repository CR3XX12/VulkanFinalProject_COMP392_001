#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera(glm::vec3 position = {0.0f, 0.0f, 0.0f}, float yaw = 0.0f, float pitch = 0.0f, 
           float aspect = 1.0f, float fov = 45.0f, float nearClip = 0.1f, float farClip = 100.0f)
        : position(position), yaw(yaw), pitch(pitch), aspect(aspect), fov(fov), nearClip(nearClip), farClip(farClip) {}

    // Rotate camera by given yaw and pitch offsets (in radians)
    void rotate(float yawOffset, float pitchOffset) {
        yaw   += yawOffset;
        pitch += pitchOffset;
        // Clamp pitch to avoid flipping (±89 degrees max)
        if (pitch >  1.55334f) pitch =  1.55334f;
        if (pitch < -1.55334f) pitch = -1.55334f;
        // Keep yaw in range [-π, π] for numerical stability
        if (yaw > 3.14159f)       yaw -= 6.28318f;
        else if (yaw < -3.14159f) yaw += 6.28318f;
    }

    // Move camera position in its local axes (W/S forward-back, A/D left-right)
    void moveForward(float distance) {
        // Forward movement on the ground plane (ignore pitch component for movement direction)
        glm::vec3 forwardDir = glm::normalize(glm::vec3(cos(yaw), sin(yaw), 0.0f));
        position += forwardDir * distance;
    }
    void moveRight(float distance) {
        glm::vec3 forwardDir = glm::normalize(glm::vec3(cos(yaw), sin(yaw), 0.0f));
        glm::vec3 rightDir = glm::normalize(glm::cross(forwardDir, glm::vec3(0.0f, 0.0f, 1.0f)));
        position += rightDir * distance;
    }

    glm::mat4 getViewMatrix() const {
        // Calculate the direction vector from yaw and pitch
        glm::vec3 direction;
        direction.x = cos(pitch) * cos(yaw);
        direction.y = cos(pitch) * sin(yaw);
        direction.z = sin(pitch);
        glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);  // world up direction
        return glm::lookAt(position, position + direction, up);
    }

    glm::mat4 getProjectionMatrix() const {
        glm::mat4 proj = glm::perspective(glm::radians(fov), aspect, nearClip, farClip);
        proj[1][1] *= -1;  // invert Y for Vulkan's coordinate system
        return proj;
    }

    glm::vec3 getPosition() const { return position; }

private:
    glm::vec3 position;
    float yaw;    // rotation around Z axis (in radians)
    float pitch;  // rotation around X axis (in radians)
    float aspect;
    float fov;
    float nearClip;
    float farClip;
};
