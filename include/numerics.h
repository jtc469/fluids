#pragma once

#include <vector>

void set_bnd(int b, std::vector<float>& x, int N);

void lin_solve0(int b, std::vector<float>& x, const std::vector<float>& x0, float a, float c, int N, int iters);

void lin_solve1(int b, std::vector<float>& x, const std::vector<float>& x0, float a, float c, int N, int iters);
