#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <vector>
#include "numerics.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

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
    Solve1,
    Solve3,
    Solve4
};

// 'Diffuse' x by a factor of a
// a = (diff * N^2) * dt
static void diffuse(int b, std::vector<float>& x, const std::vector<float>& x0,
                    float diff, float dt, int N, LinSolver solver,
                    int lin_solve_iters, std::vector<float>& lin_solve_scratch) {
    float a = dt * diff * N * N;
    x = x0;

    switch (solver) {
        case LinSolver::Solve0:
            lin_solve0(b, x, x0, a, 1.0f + 4.0f * a, N, lin_solve_iters);
            break;
        case LinSolver::Solve1:
            lin_solve1(b, x, x0, a, 1.0f + 4.0f * a, N, lin_solve_iters, lin_solve_scratch);
            break;
        case LinSolver::Solve3:
            lin_solve3(b, x, x0, a, 1.0f + 4.0f * a, N, lin_solve_iters);
            break;
        case LinSolver::Solve4:
            lin_solve4(b, x, x0, a, 1.0f + 4.0f * a, N, lin_solve_iters);
            break;
    }
}


// Trace each cell backwards, copy value that cell at that previous position.
static void transport(int b, std::vector<float>& d, const std::vector<float>& d0, const std::vector<float>& u, const std::vector<float>& v, float dt, int N) {
    const int stride = N + 2;
    const float dt0 = dt * N; // scale timestep by resolution
    const float lo = 0.5f;
    const float hi = (float)N + 0.5f;

#ifdef USE_OMP_TARGET
    const int sz = (int)d.size();
    float* dp = d.data();
    const float* d0p = d0.data();
    const float* up = u.data();
    const float* vp = v.data();

    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: d0p[0:sz], up[0:sz], vp[0:sz]) map(from: dp[0:sz])
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            const int id = j * stride + i;
            float x = (float)i - dt0 * up[id];
            float y = (float)j - dt0 * vp[id];
            x = x < lo ? lo : (x > hi ? hi : x);
            y = y < lo ? lo : (y > hi ? hi : y);

            const int i0 = (int)x;
            const int j0 = (int)y;
            const int i1 = i0 + 1;
            const int j1 = j0 + 1;

            const float sx = x - (float)i0;
            const float sy = y - (float)j0;

            const int row0 = j0 * stride;
            const int row1 = j1 * stride;
            const float c00 = d0p[row0 + i0];
            const float c10 = d0p[row0 + i1];
            const float c01 = d0p[row1 + i0];
            const float c11 = d0p[row1 + i1];

            const float x0 = c00 + sx * (c10 - c00);
            const float x1 = c01 + sx * (c11 - c01);
            dp[id] = x0 + sy * (x1 - x0);
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        const int row = j * stride;
        for (int i = 1; i <= N; i++) {
            const int id = row + i;

            // where did density at (i,j) come from dt ago?
            float x = (float)i - dt0 * u[id];
            float y = (float)j - dt0 * v[id];
            x = std::max(lo, std::min(x, hi));
            y = std::max(lo, std::min(y, hi));

            const int i0 = (int)x;
            const int j0 = (int)y;
            const int i1 = i0 + 1;
            const int j1 = j0 + 1;

            const float sx = x - (float)i0;
            const float sy = y - (float)j0;

            const int row0 = j0 * stride;
            const int row1 = j1 * stride;
            const float c00 = d0[row0 + i0];
            const float c10 = d0[row0 + i1];
            const float c01 = d0[row1 + i0];
            const float c11 = d0[row1 + i1];

            const float x0 = c00 + sx * (c10 - c00);
            const float x1 = c01 + sx * (c11 - c01);
            d[id] = x0 + sy * (x1 - x0);
        }
    }
#endif
    set_bnd(b, d, N);
}

// decay x by 1 + a *dt
static void dissipate(std::vector<float>& x, float a, float dt, int N) {
    if (a <= 0.0f) return;
    const int stride = N + 2;
    const float inv_D = 1.0f / (1.0f + a * dt);

#ifdef USE_OMP_TARGET
    const int sz = (int)x.size();
    float* xp = x.data();
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: xp[0:sz])
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            xp[j * stride + i] *= inv_D;
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        const int row = j * stride;
        for (int i = 1; i <= N; i++) {
            x[row + i] *= inv_D;
        }
    }
#endif
    set_bnd(0, x, N);
}

static void project(std::vector<float>& u, std::vector<float>& v, std::vector<float>& p,
                    std::vector<float>& div, int N, LinSolver solver,
                    int lin_solve_iters, std::vector<float>& lin_solve_scratch) {
    const int stride = N + 2;
    const float h = 1.0f / N;
    const float scale_div = -0.5f * h;

    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        const int row = j * stride;
        const int row_m = row - stride;
        const int row_p = row + stride;
        for (int i = 1; i <= N; i++) {
            const int id = row + i;
            div[id] = scale_div *
                      (u[id + 1] - u[id - 1] +
                       v[row_p + i] - v[row_m + i]);
            p[id] = 0.0f;
        }
    }
    set_bnd(0, div, N);
    set_bnd(0, p, N);

    switch (solver) {
        case LinSolver::Solve0:
            lin_solve0(0, p, div, 1.0f, 4.0f, N, lin_solve_iters);
            break;
        case LinSolver::Solve1:
            lin_solve1(0, p, div, 1.0f, 4.0f, N, lin_solve_iters, lin_solve_scratch);
            break;
        case LinSolver::Solve3:
            lin_solve3(0, p, div, 1.0f, 4.0f, N, lin_solve_iters);
            break;
        case LinSolver::Solve4:
            lin_solve4(0, p, div, 1.0f, 4.0f, N, lin_solve_iters);
            break;
    }

    const float grad_scale = 0.5f / h;
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        const int row = j * stride;
        const int row_m = row - stride;
        const int row_p = row + stride;
        for (int i = 1; i <= N; i++) {
            const int id = row + i;
            u[id] -= grad_scale * (p[id + 1] - p[id - 1]);
            v[id] -= grad_scale * (p[row_p + i] - p[row_m + i]);
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
        std::vector<float> lin_solve_scratch;

        Fluid(int n, float dt_, float visc_, float diff_, float diss_, LinSolver solver_,
                    int lin_solve_iters_)
                : N(n), lin_solve_iters(lin_solve_iters_), dt(dt_), visc(visc_), diff(diff_),
                    diss(diss_), solver(solver_) {
        int sz = (N + 2) * (N + 2);
        u.assign(sz, 0.0f);
        v.assign(sz, 0.0f);
        u0.assign(sz, 0.0f);
        v0.assign(sz, 0.0f);
        dens.assign(sz, 0.0f);
        dens0.assign(sz, 0.0f);
        p.assign(sz, 0.0f);
        div.assign(sz, 0.0f);
        lin_solve_scratch.assign(sz, 0.0f);
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

        diffuse(1, u0, u, visc, dt, N, solver, lin_solve_iters, lin_solve_scratch);
        diffuse(2, v0, v, visc, dt, N, solver, lin_solve_iters, lin_solve_scratch);
        project(u0, v0, p, div, N, solver, lin_solve_iters, lin_solve_scratch);

        transport(1, u, u0, u0, v0, dt, N);
        transport(2, v, v0, u0, v0, dt, N);
        project(u, v, p, div, N, solver, lin_solve_iters, lin_solve_scratch);

    }

    void Sstep() {

        diffuse(0, dens0, dens, diff, dt, N, solver, lin_solve_iters, lin_solve_scratch);
        transport(0, dens, dens0, u, v, dt, N);
        dissipate(dens, diss, dt, N);

    }

    void step() {
        Vstep();
        Sstep();
    }
};

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
    const int mpi_rank = 0;
    const int mpi_size = 1;
#endif

    int N = 128;
    int steps = 300;
    int write_every = 4;
    int inject_steps = (int)steps*3/4;
    int lin_solve_iters = 24;
    LinSolver solver = LinSolver::Solve3;

    /*
        FLUID PARAMETERS
        viscosity=1e-4f, diffusivity=1e-4f, dissipation=1.0f
    */
    float dt = 0.1f;
    float visc = 1e-6f; // 1e-4f
    float diff = 1e-6f; // 1e-4f
    float diss = 0.02f; // 0.02f

    if (argc >= 2) N = std::max(16, std::atoi(argv[1]));
    if (argc >= 3) steps = std::max(1, std::atoi(argv[2]));

    if (argc >= 4) visc = std::atof(argv[3]);
    if (argc >= 5) diff = std::atof(argv[4]);
    if (argc >= 6) diss = std::atof(argv[5]);
    if (argc >= 7) lin_solve_iters = std::max(1, std::atoi(argv[6]));
    if (argc >= 8) {
        int solver_id = std::atoi(argv[7]);
        if (solver_id == 0) solver = LinSolver::Solve0;
        else if (solver_id == 1) solver = LinSolver::Solve1;
        else if (solver_id == 3) solver = LinSolver::Solve3;
        else if (solver_id == 4) solver = LinSolver::Solve4;
    }

    Fluid fluid(N, dt, visc, diff, diss, solver, lin_solve_iters);

    auto start = std::chrono::steady_clock::now();

    std::string out_path = "build/density.bin";
#ifdef USE_MPI
    if (mpi_size > 1) {
        out_path = "build/density_rank" + std::to_string(mpi_rank) + ".bin";
    }
#endif
    std::ofstream out(out_path, std::ios::binary);
    write_i32(out, (int32_t)N);
    int frames_out = (steps + write_every - 1) / write_every;
    write_i32(out, (int32_t)frames_out);
    int32_t sz = (int32_t)((N + 2) * (N + 2));
    write_i32(out, sz);

    const int spawn_size = std::max(2, N / 16);
    const int start_x = (N * (mpi_rank + 1)) / (mpi_size + 1);
    const int start_y = N / 6;
    const float sigma = 1.6f; // source softness radius (larger = smoother/wider injector)
    const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);

    std::vector<int> source_ids;
    std::vector<float> source_w;
    source_ids.reserve((2 * spawn_size + 1) * (2 * spawn_size + 1) * 4);
    source_w.reserve((2 * spawn_size + 1) * (2 * spawn_size + 1) * 4);

    for (int j = -spawn_size; j <= spawn_size; j++) {
        for (int i = -spawn_size; i <= spawn_size; i++) {
            const float r2 = (float)(i * i + j * j);
            const float w = std::exp(-r2 * inv_two_sigma2);

            const int sx = start_x + i;
            const int sy = start_y + j;
            if (sx < 1 || sx > N || sy < 1 || sy > N) continue;
            source_ids.push_back(IX(sx, sy, N));
            source_w.push_back(w);
        }
    }
    

    for (int t = 0; t < steps; t++) {

        if (t < inject_steps) {
            const size_t nsrc = source_ids.size();
            for (size_t s = 0; s < nsrc; s++) {
                const int id = source_ids[s];
                const float w = source_w[s];
                fluid.dens[id] += 0.1f * w;
                fluid.u[id] += 2.0f * w;
                fluid.v[id] += 5.0f * w;
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

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}