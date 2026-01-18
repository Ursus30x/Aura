#include <glad/gl.h> 
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "objLoader.h"

// --- Morphing Vertex Shader ---
const char* vertexShaderSource = R"(
#version 450 core
// Model 1 Data
layout (location = 0) in vec3 aPos1;
layout (location = 1) in vec3 aNormal1;
layout (location = 2) in vec2 aTexCoord; // We use UVs from Model 1 only

// Model 2 Data (Target)
layout (location = 3) in vec3 aPos2;
layout (location = 4) in vec3 aNormal2;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float blendFactor; // 0.0 = Model 1, 1.0 = Model 2

void main() {
    // Linear Interpolation (Lerp)
    vec3 mixedPos = mix(aPos1, aPos2, blendFactor);
    vec3 mixedNorm = normalize(mix(aNormal1, aNormal2, blendFactor));

    FragPos = vec3(model * vec4(mixedPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * mixedNorm;  
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 450 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPos;

void main() {
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
  
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

GLuint CreateShader(const char* vSource, const char* fSource) {
    auto compile = [](GLenum type, const char* src) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, NULL);
        glCompileShader(shader);
        return shader;
    };
    GLuint vs = compile(GL_VERTEX_SHADER, vSource);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fSource);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main(int argc, char** argv) {
    // Check args
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model1.obj> <model2.obj>" << std::endl;
        return -1;
    }

    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Morph Target Test", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (gladLoadGL(glfwGetProcAddress) == 0) return -1;

    glViewport(0, 0, 1280, 720);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    // --- LOAD MODELS ---
    std::vector<Vertex> mesh1, mesh2;
    bool loaded1 = loadOBJ(argv[1], mesh1);
    bool loaded2 = loadOBJ(argv[2], mesh2);

    if (!loaded1 || !loaded2) {
        std::cerr << "Failed to load one of the models." << std::endl;
        return -1;
    }

    // CRITICAL: Topology check
    // Vertex count must match exactly for morphing to work by index
    if (mesh1.size() != mesh2.size()) {
        std::cerr << "Error: Models must have identical vertex counts!" << std::endl;
        std::cerr << "Model 1: " << mesh1.size() << " | Model 2: " << mesh2.size() << std::endl;
        return -1;
    }

    // --- GPU BUFFERS ---
    GLuint VAO, VBO1, VBO2;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO1);
    glGenBuffers(1, &VBO2);

    glBindVertexArray(VAO);

    // 1. Bind Model 1 Data (Positions, Normals, UVs)
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, mesh1.size() * sizeof(Vertex), mesh1.data(), GL_STATIC_DRAW);

    // Attrib 0: Pos1
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    // Attrib 1: Normal1
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(1);
    // Attrib 2: UV (Shared)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texUV));
    glEnableVertexAttribArray(2);

    // 2. Bind Model 2 Data (Positions, Normals)
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, mesh2.size() * sizeof(Vertex), mesh2.data(), GL_STATIC_DRAW);

    // Attrib 3: Pos2
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(3);
    // Attrib 4: Normal2
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(4);

    GLuint shaderProgram = CreateShader(vertexShaderSource, fragmentShaderSource);
    
    // State
    glEnable(GL_DEPTH_TEST);
    glm::vec3 clear_color(0.2f);
    glm::vec3 mesh_color(1.0f, 0.5f, 0.2f);
    glm::vec3 camera_view(0.0f, 0.0f, -4.0f);
    
    // Rotation state
    bool autoRotate = true;
    float rotationSpeed = 1.0f;
    float manualRotation = 0.0f; // In degrees
    float blendFactor = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Morph Controls"); 
            ImGui::Text("Vertices: %lu", mesh1.size());
            ImGui::Separator();
            
            // Morph Slider
            ImGui::SliderFloat("Morph Blend", &blendFactor, 0.0f, 1.0f);
            ImGui::Separator();

            // Rotation Controls
            ImGui::Checkbox("Auto Rotate", &autoRotate);
            if (autoRotate) {
                ImGui::SliderFloat("Speed", &rotationSpeed, 0.0f, 5.0f);
            } else {
                // Slider from 0 to 360 degrees
                ImGui::SliderFloat("Angle", &manualRotation, 0.0f, 360.0f);
            }
            
            ImGui::Separator();
            ImGui::ColorEdit3("Mesh Color", (float*)&mesh_color);
            ImGui::DragFloat3("Camera", (float*)&camera_view, 0.1f);
            ImGui::End();
        }

        ImGui::Render();
        
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Uniforms
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)w / (float)h, 0.1f, 100.0f);
        glm::mat4 view = glm::translate(glm::mat4(1.0f), camera_view);
        glm::mat4 model = glm::mat4(1.0f);

        // --- ROTATION LOGIC ---
        float finalAngle;
        if (autoRotate) {
            finalAngle = (float)glfwGetTime() * rotationSpeed;
        } else {
            finalAngle = glm::radians(manualRotation);
        }
        model = glm::rotate(model, finalAngle, glm::vec3(0.0f, 1.0f, 0.0f));
        // ----------------------

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        
        glUniform1f(glGetUniformLocation(shaderProgram, "blendFactor"), blendFactor);
        glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, &mesh_color[0]);
        glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 2.0f, 2.0f, 2.0f);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, mesh1.size());

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO1);
    glDeleteBuffers(1, &VBO2);
    glDeleteProgram(shaderProgram);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}