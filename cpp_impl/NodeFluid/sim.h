#pragma once

#include <vector>
#include <glm/glm.hpp>

void init();
bool fit();
void step();
std::vector<float>& getVelocityX();
std::vector<float>& getVelocityY();
std::vector<float>& getVelocityZ();
std::vector<float>& getNodeVelocityX();
std::vector<float>& getNodeVelocityY();
std::vector<float>& getNodeVelocityZ();