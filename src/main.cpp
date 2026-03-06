#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>
#include "numerics.h"

// Binary dumper functions
static void write_i32(std::ofstream& out, int32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
static void write_f32_block(std::ofstream& out, const std::vector<float>& a) {
    out.write(reinterpret_cast<const char*>(a.data()),
              (std::streamsize)(a.size() * sizeof(float)));
}

// Index flattener with ghost region
static inline int IX(int i, int j, int N) { return i + (N + 2) * j; }

enum class LinSolver {
    Solve0,
    Solve1
};

// 'Diffuse' x by a factor of a
// a = (diff * N^2) * dt
static void diffuse(int b, std::vector<float>& x, const std::vector<float>& x0,
                    float diff, float dt, int N, LinSolver solver) {
    float a = dt * diff * N * N;
    x = x0;

    if (solver == LinSolver::Solve0) {
        lin_solve0(b, x, x0, a, 1.0f + 4.0f * a, N, 24);
    } else {
        lin_solve1(b, x, x0, a, 1.0f + 4.0f * a, N, 24);
    }
}


// Bilinear sample around (x, y) in cell-center coordinates.
// This replaces nearest-neighbor sampling to reduce blocky advection artifacts.
static float sample_bilinear(const std::vector<float>& d0, float x, float y, int N) {
    x = std::clamp(x, 0.5f, (float)N + 0.5f);
    y = std::clamp(y, 0.5f, (float)N + 0.5f);

    int i0 = (int)std::floor(x);
    int j0 = (int)std::floor(y);
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    i0 = std::clamp(i0, 0, N + 1);
    j0 = std::clamp(j0, 0, N + 1);
    i1 = std::clamp(i1, 0, N + 1);
    j1 = std::clamp(j1, 0, N + 1);

    float sx = x - (float)i0;
    float sy = y - (float)j0;

    float c00 = d0[IX(i0, j0, N)];
    float c10 = d0[IX(i1, j0, N)];
    float c01 = d0[IX(i0, j1, N)];
    float c11 = d0[IX(i1, j1, N)];

    float start_x0 = c00 + sx * (c10 - c00);
    float start_x1 = c01 + sx * (c11 - c01);
    return start_x0 + sy * (start_x1 - start_x0);
}

// Trace each cell backwards, copy value that cell at that previous position.
static void transport(int b, std::vector<float>& d, const std::vector<float>& d0, const std::vector<float>& u, const std::vector<float>& v, float dt, int N) {
    float dt0 = dt * N; // scale timestep by resolution
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            int id = IX(i, j, N);

            // where did density at (i,j) come from dt ago?
            float x = (float)i - dt0 * u[id]; 
            float y = (float)j - dt0 * v[id];
            d[id] = sample_bilinear(d0, x, y, N); // smoother backtrace sample
        }
    }
    set_bnd(b, d, N);
}

// decay x by 1 + a *dt
static void dissipate(std::vector<float>& x, float a, float dt, int N) {
    if (a <= 0.0f) return;
    float D = 1.0f + a * dt;

    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            x[IX(i, j, N)] /= D;
        }
    }
    set_bnd(0, x, N);
}

static void project(std::vector<float>& u, std::vector<float>& v, std::vector<float>& p, std::vector<float>& div, int N, LinSolver solver) {
    float h = 1.0f / N;

    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            div[IX(i, j, N)] = -0.5f * h *
                               (u[IX(i + 1, j, N)] - u[IX(i - 1, j, N)] +
                                v[IX(i, j + 1, N)] - v[IX(i, j - 1, N)]);
            p[IX(i, j, N)] = 0.0f;
        }
    }
    set_bnd(0, div, N);
    set_bnd(0, p, N);

    if (solver == LinSolver::Solve0) {
        lin_solve0(0, p, div, 1.0f, 4.0f, N, 24);
    } else {
        lin_solve1(0, p, div, 1.0f, 4.0f, N, 24);
    }

    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            u[IX(i, j, N)] -= 0.5f * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]) / h;
            v[IX(i, j, N)] -= 0.5f * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]) / h;
        }
    }
    set_bnd(1, u, N);
    set_bnd(2, v, N);
}

struct Fluid {
    int N;
    int lin_solve_iters;
    float dt;
    float visc;
    float diff;
    float diss;
    LinSolver solver;

    std::vector<float> u, v, u0, v0;
    std::vector<float> dens, dens0;
    std::vector<float> p, div;

    Fluid(int n, float dt_, float visc_, float diff_, float diss_, LinSolver solver_): N(n), dt(dt_), visc(visc_), diff(diff_), diss(diss_), solver(solver_) {
        int sz = (N + 2) * (N + 2);
        u.assign(sz, 0.0f);
        v.assign(sz, 0.0f);
        u0.assign(sz, 0.0f);
        v0.assign(sz, 0.0f);
        dens.assign(sz, 0.0f);
        dens0.assign(sz, 0.0f);
        p.assign(sz, 0.0f);
        div.assign(sz, 0.0f);
    }

    void addDensity(int i, int j, float a) {
        if (i < 1 || i > N || j < 1 || j > N) return;
        dens[IX(i, j, N)] += a;
    }

    void addVelocity(int i, int j, float ax, float ay) {
        if (i < 1 || i > N || j < 1 || j > N) return;
        int id = IX(i, j, N);
        u[id] += ax;
        v[id] += ay;
    }

    void Vstep() {

        diffuse(1, u0, u, visc, dt, N, solver);
        diffuse(2, v0, v, visc, dt, N, solver);
        project(u0, v0, p, div, N, solver);

        transport(1, u, u0, u0, v0, dt, N);
        transport(2, v, v0, u0, v0, dt, N);
        project(u, v, p, div, N, solver);

    }

    void Sstep() {

        diffuse(0, dens0, dens, diff, dt, N, solver);
        transport(0, dens, dens0, u, v, dt, N);
        dissipate(dens, diss, dt, N);

    }

    void step() {
        Vstep();
        Sstep();
    }
};

int main(int argc, char** argv) {
    int N = 128;
    int steps = 300;
    int write_every = 4;
    int inject_steps = (int)steps*3/4;
    int lin_solve_iters = 24;
    LinSolver solver = LinSolver::Solve1;

    /*
        FLUID PARAMETERS
        viscosity=1e-4f, diffusivity=1e-4f, dissipation=1.0f
    */
    float dt = 0.1f;
    float visc = 1e-6f; // 1e-4f
    float diff = 1e-6f; // 1e-4f
    float diss = 0.0f; // 0.0f

    if (argc >= 2) N = std::max(16, std::atoi(argv[1]));
    if (argc >= 3) steps = std::max(1, std::atoi(argv[2]));

    if (argc >= 4) visc = std::atof(argv[3]);
    if (argc >= 5) diff = std::atof(argv[4]);
    if (argc >= 6) diss = std::atof(argv[5]);

    Fluid fluid(N, dt, visc, diff, diss, solver);

    auto start = std::chrono::steady_clock::now();

    std::ofstream out("build/density.bin", std::ios::binary);
    write_i32(out, (int32_t)N);
    int frames_out = (steps + write_every - 1) / write_every;
    write_i32(out, (int32_t)frames_out);
    int32_t sz = (int32_t)((N + 2) * (N + 2));
    write_i32(out, sz);

    int spawn_size = std::max(2, N / 16);
    

    for (int t = 0; t < steps; t++) {

        if (t < inject_steps) {
            int start_x = N / 2;
            
            int start_y = N / 6;
            float sigma = 1.6; // source softness radius (larger = smoother/wider injector)
            float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);

            for (int j = -spawn_size; j <= spawn_size; j++) {
                for (int i = -spawn_size; i <= spawn_size; i++) {
                    float r2 = (float)(i * i + j * j);
                    float w = std::exp(-r2 * inv_two_sigma2); // gaussian falloff from source center

                    fluid.addDensity(start_x + i, start_y + j, 0.1f * w);
                    fluid.addVelocity(start_x + i, start_y + j, 0.0f * w, 5.0f * w);
                }
            }
        }

        fluid.step();
        if ((t % write_every) == 0) {
            write_f32_block(out, fluid.dens);
        }
        int simmed = t + 1;
        if ((simmed % 100) == 0 || simmed == steps) {
            std::cout << "Progress: " << simmed << "/" << steps << " steps\n";
        }
    }
    out.close();
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Sim complete: N=" << N << ", steps=" << steps << ", took "
              << (elapsed_ms / 1000.0) << " s\n";

    return 0;
}