#include "objLoader.h"
#include <iostream>
#include <cstdio>
#include <cstring>

bool loadOBJ(const char* path, std::vector<Vertex>& out_vertices) {
    std::vector<glm::vec3> temp_positions;
    std::vector<glm::vec3> temp_normals;
    std::vector<glm::vec2> temp_uvs;

    FILE* file = fopen(path, "r");
    if (file == NULL) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    char line[1024];
    while (fgets(line, 1024, file)) {
        // Vertex Information
        if (strncmp(line, "v ", 2) == 0) {
            glm::vec3 v;
            sscanf(line + 2, "%f %f %f", &v.x, &v.y, &v.z);
            temp_positions.push_back(v);
        }
        else if (strncmp(line, "vt ", 3) == 0) {
            glm::vec2 uv;
            sscanf(line + 3, "%f %f", &uv.x, &uv.y);
            temp_uvs.push_back(uv);
        }
        else if (strncmp(line, "vn ", 3) == 0) {
            glm::vec3 n;
            sscanf(line + 3, "%f %f %f", &n.x, &n.y, &n.z);
            temp_normals.push_back(n);
        }
        // Face Information
        else if (strncmp(line, "f ", 2) == 0) {
            unsigned int vIdx[4], uvIdx[4], nIdx[4];
            int matches;

            // Zero out arrays to be safe
            memset(vIdx, 0, sizeof(vIdx));
            memset(uvIdx, 0, sizeof(uvIdx));
            memset(nIdx, 0, sizeof(nIdx));

            // Attempt to read as a QUAD (4 vertices) in format v/vt/vn
            matches = sscanf(line + 2, 
                "%d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d",
                &vIdx[0], &uvIdx[0], &nIdx[0],
                &vIdx[1], &uvIdx[1], &nIdx[1],
                &vIdx[2], &uvIdx[2], &nIdx[2],
                &vIdx[3], &uvIdx[3], &nIdx[3]
            );

            int vertices_in_face = 0;
            
            // Logic to determine if Quad or Triangle based on sscanf matches
            // 12 matches = Quad (v/vt/vn * 4)
            // 9 matches  = Triangle (v/vt/vn * 3)
            if (matches == 12) vertices_in_face = 4;
            else if (matches == 9) vertices_in_face = 3;
            else {
                // Try format v//vn (no UVs)
                matches = sscanf(line + 2, 
                    "%d//%d %d//%d %d//%d %d//%d",
                    &vIdx[0], &nIdx[0],
                    &vIdx[1], &nIdx[1],
                    &vIdx[2], &nIdx[2],
                    &vIdx[3], &nIdx[3]
                );
                if (matches == 8) vertices_in_face = 4;
                else if (matches == 6) vertices_in_face = 3;
            }

            // If we still haven't found it, try just v
            if (vertices_in_face == 0) {
                matches = sscanf(line + 2, "%d %d %d %d", &vIdx[0], &vIdx[1], &vIdx[2], &vIdx[3]);
                if (matches == 4) vertices_in_face = 4;
                else if (matches == 3) vertices_in_face = 3;
            }
            
            // Define the triangles indices: 
            // Tri 1: 0, 1, 2
            // Tri 2: 0, 2, 3 (Only if quad)
            int indicesToProcess[] = {0, 1, 2, 0, 2, 3};
            int numVerticesToCreate = (vertices_in_face == 4) ? 6 : 3;

            for (int i = 0; i < numVerticesToCreate; i++) {
                int idx = indicesToProcess[i];

                Vertex v;
                // OBJ indices are 1-based, C++ is 0-based
                v.position = temp_positions[vIdx[idx] - 1];
                
                if (uvIdx[idx] != 0) v.texUV = temp_uvs[uvIdx[idx] - 1];
                else v.texUV = glm::vec2(0.0f);

                if (nIdx[idx] != 0) v.normal = temp_normals[nIdx[idx] - 1];
                else v.normal = glm::vec3(0.0f, 0.0f, 1.0f);

                out_vertices.push_back(v);
            }
        }
    }

    fclose(file);
    return true;
}