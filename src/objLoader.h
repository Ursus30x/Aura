#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texUV;
};

bool loadOBJ(const char* path, std::vector<Vertex>& out_vertices);