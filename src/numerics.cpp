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
                float a, float c, int N, int iters) {
    const float eps = 1e-3f;
    std::vector<float> x_new = x;

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