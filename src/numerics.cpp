#include "numerics.h"
#include <cmath>


// Index flattener with ghost region
static inline int IX(int i, int j, int N) { return i + (N + 2) * j; }


/*
    Updates x, applying boundary conditions at ghost regions
    
    b==0: scalar fields, no flip
    b==1: ux flips
    b==2: uy flips

*/
void set_bnd(int b, std::vector<float>& x, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= N; i++) {
        if (b == 1) {
            x[IX(0, i, N)] = -x[IX(1, i, N)];
            x[IX(N + 1, i, N)] = -x[IX(N, i, N)];
        } else {
            x[IX(0, i, N)] = x[IX(1, i, N)];
            x[IX(N + 1, i, N)] = x[IX(N, i, N)];
        }

        if (b == 2) {
            x[IX(i, 0, N)] = -x[IX(i, 1, N)];
            x[IX(i, N + 1, N)] = -x[IX(i, N, N)];
        } else {
            x[IX(i, 0, N)] = x[IX(i, 1, N)];
            x[IX(i, N + 1, N)] = x[IX(i, N, N)];
        }
    }

    // Corner values are averaged from adjacent ghost edges for stability.
    x[IX(0, 0, N)] = 0.5f * (x[IX(1, 0, N)] + x[IX(0, 1, N)]);
    x[IX(0, N + 1, N)] = 0.5f * (x[IX(1, N + 1, N)] + x[IX(0, N, N)]);
    x[IX(N + 1, 0, N)] = 0.5f * (x[IX(N, 0, N)] + x[IX(N + 1, 1, N)]);
    x[IX(N + 1, N + 1, N)] = 0.5f * (x[IX(N, N + 1, N)] + x[IX(N + 1, N, N)]);
}



/*
lin_solve0: fast linear-system approximate for a vector<float> x
x = (x0 + a * neighbors) / c
4 neighbors, a is scaled by dt and grid size
*/
void lin_solve0(int b, std::vector<float>& x, const std::vector<float>& x0, float a, float c, int N, int iters) {
    const float eps = 1e-3f;

    for (int it = 0; it < iters; it++) {
        float max_delta = 0.0f;

        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                int id = IX(i, j, N);

                float upd =
                    (x0[id] +
                     a * (x[IX(i - 1, j, N)] + x[IX(i + 1, j, N)] +
                          x[IX(i, j - 1, N)] + x[IX(i, j + 1, N)])) / c;

                float delta = std::abs(upd - x[id]);
                if (delta > max_delta) max_delta = delta;

                x[id] = upd;
            }
        }

        set_bnd(b, x, N);

        if (max_delta < eps) {
            break;
        }
    }
}

// lin_solve1 a parallelised solver  
void lin_solve1(int b, std::vector<float>& x, const std::vector<float>& x0,
                float a, float c, int N, int iters,
                std::vector<float>& scratch) {
    const float eps = 1e-3f;
    if (scratch.size() != x.size()) {
        scratch.assign(x.size(), 0.0f);
    }
    std::vector<float>& x_new = scratch;

    for (int it = 0; it < iters; it++) {
        float max_delta = 0.0f;

        #pragma omp parallel for reduction(max:max_delta) schedule(static)
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                int id = IX(i, j, N);

                float upd =
                    (x0[id] +
                     a * (x[IX(i - 1, j, N)] + x[IX(i + 1, j, N)] +
                          x[IX(i, j - 1, N)] + x[IX(i, j + 1, N)])) / c;

                float delta = std::abs(upd - x[id]);
                if (delta > max_delta) max_delta = delta;

                x_new[id] = upd;
            }
        }

        x.swap(x_new);
        set_bnd(b, x, N);

        if (max_delta < eps) {
            break;
        }
    }
}

// lin_solve3: parallel red-black Gauss-Seidel.
// In-place update reduces memory traffic vs Jacobi and converges faster per iteration.
void lin_solve3(int b, std::vector<float>& x, const std::vector<float>& x0,
                float a, float c, int N, int iters) {
    const float eps = 1e-3f;
    const int stride = N + 2;
    const float inv_c = 1.0f / c;

    for (int it = 0; it < iters; it++) {
        float max_delta = 0.0f;

        #pragma omp parallel
        {
            float thread_max = 0.0f;

            for (int color = 0; color < 2; color++) {
                #pragma omp for schedule(static) nowait
                for (int j = 1; j <= N; j++) {
                    const int row = j * stride;
                    const int i_start = 1 + ((j + color) & 1);
                    for (int i = i_start; i <= N; i += 2) {
                        const int id = row + i;
                        const float upd =
                            (x0[id] +
                             a * (x[id - 1] + x[id + 1] +
                                  x[id - stride] + x[id + stride])) * inv_c;

                        const float delta = std::abs(upd - x[id]);
                        if (delta > thread_max) thread_max = delta;
                        x[id] = upd;
                    }
                }

                #pragma omp barrier
            }

            #pragma omp critical
            {
                if (thread_max > max_delta) max_delta = thread_max;
            }
        }

        set_bnd(b, x, N);

        if (max_delta < eps) {
            break;
        }
    }
}

static void rb_smooth(int b, std::vector<float>& x, const std::vector<float>& rhs,
                      float a, float c, int N, int iters) {
    const int stride = N + 2;
    const float inv_c = 1.0f / c;

    for (int it = 0; it < iters; it++) {
        #pragma omp parallel
        {
            for (int color = 0; color < 2; color++) {
                #pragma omp for schedule(static) nowait
                for (int j = 1; j <= N; j++) {
                    const int row = j * stride;
                    const int i_start = 1 + ((j + color) & 1);
                    for (int i = i_start; i <= N; i += 2) {
                        const int id = row + i;
                        x[id] =
                            (rhs[id] +
                             a * (x[id - 1] + x[id + 1] +
                                  x[id - stride] + x[id + stride])) * inv_c;
                    }
                }
                #pragma omp barrier
            }
        }
        set_bnd(b, x, N);
    }
}

static void compute_residual(std::vector<float>& res, const std::vector<float>& x,
                             const std::vector<float>& rhs, float a, float c, int N) {
    const int stride = N + 2;

    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; j++) {
        const int row = j * stride;
        for (int i = 1; i <= N; i++) {
            const int id = row + i;
            const float Ax = c * x[id] - a * (x[id - 1] + x[id + 1] + x[id - stride] + x[id + stride]);
            res[id] = rhs[id] - Ax;
        }
    }
    set_bnd(0, res, N);
}

static void restrict_full_weight(std::vector<float>& coarse, const std::vector<float>& fine, int Nf) {
    const int Nc = Nf / 2;
    const int sf = Nf + 2;
    const int sc = Nc + 2;

    std::fill(coarse.begin(), coarse.end(), 0.0f);

    #pragma omp parallel for schedule(static)
    for (int jc = 1; jc <= Nc; jc++) {
        const int jf = 2 * jc;
        for (int ic = 1; ic <= Nc; ic++) {
            const int ifn = 2 * ic;
            const int cid = ic + sc * jc;
            const int fid = ifn + sf * jf;

            coarse[cid] =
                0.25f * fine[fid] +
                0.125f * (fine[fid - 1] + fine[fid + 1] + fine[fid - sf] + fine[fid + sf]) +
                0.0625f * (fine[fid - sf - 1] + fine[fid - sf + 1] + fine[fid + sf - 1] + fine[fid + sf + 1]);
        }
    }
    set_bnd(0, coarse, Nc);
}

static void prolong_add(std::vector<float>& fine, const std::vector<float>& coarse, int Nc) {
    const int Nf = Nc * 2;
    const int sf = Nf + 2;
    const int sc = Nc + 2;

    #pragma omp parallel for schedule(static)
    for (int jc = 1; jc <= Nc; jc++) {
        const int jf = 2 * jc;
        for (int ic = 1; ic <= Nc; ic++) {
            const int ifn = 2 * ic;
            const float v = coarse[ic + sc * jc];

            fine[ifn + sf * jf] += v;
            if (ifn - 1 >= 1) fine[(ifn - 1) + sf * jf] += v;
            if (jf - 1 >= 1) fine[ifn + sf * (jf - 1)] += v;
            if (ifn - 1 >= 1 && jf - 1 >= 1) fine[(ifn - 1) + sf * (jf - 1)] += v;
        }
    }
}

static void mg_vcycle(int b, std::vector<float>& x, const std::vector<float>& rhs,
                      float a, float c, int N, int depth) {
    if (N <= 8 || depth >= 10) {
        rb_smooth(b, x, rhs, a, c, N, 16);
        return;
    }

    rb_smooth(b, x, rhs, a, c, N, 3);

    std::vector<float> residual((N + 2) * (N + 2), 0.0f);
    compute_residual(residual, x, rhs, a, c, N);

    const int Nc = N / 2;
    std::vector<float> rhs_c((Nc + 2) * (Nc + 2), 0.0f);
    std::vector<float> err_c((Nc + 2) * (Nc + 2), 0.0f);

    restrict_full_weight(rhs_c, residual, N);

    // Coarse-grid operator scaling: Laplacian strength scales with h^-2.
    const float a_c = 0.25f * a;
    mg_vcycle(0, err_c, rhs_c, a_c, c, Nc, depth + 1);

    prolong_add(x, err_c, Nc);
    set_bnd(b, x, N);
    rb_smooth(b, x, rhs, a, c, N, 2);
}

void lin_solve4(int b, std::vector<float>& x, const std::vector<float>& x0,
                float a, float c, int N, int iters) {
    const int cycles = std::max(1, iters / 4);
    for (int cyc = 0; cyc < cycles; cyc++) {
        mg_vcycle(b, x, x0, a, c, N, 0);
    }
}