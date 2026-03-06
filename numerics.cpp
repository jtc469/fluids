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

    x[IX(0, 0, N)] = 0.5f * (x[IX(1, 0, N)] + x[IX(0, 1, N)]);
    x[IX(0, N + 1, N)] = 0.5f * (x[IX(1, N + 1, N)] + x[IX(0, N, N)]);
    x[IX(N + 1, 0, N)] = 0.5f * (x[IX(N, 0, N)] + x[IX(N + 1, 1, N)]);
    x[IX(N + 1, N + 1, N)] = 0.5f * (x[IX(N, N + 1, N)] + x[IX(N + 1, N, N)]);
}


/*
lin_solve: 12 step fast linear-system approximate for a vector<float> x
    x = (x0 + a*(neighbors)) / c;

Diffusion: a = dt*diff*N*N, c = 1 + 4*a
    because we have 4 neighbors, and the diffusion rate is scaled by dt and grid size
    ρ - (dt*diff*N*N)(ρ_neighbours) = ρ0
    => ρ = (ρ0 + (dt*diff*N*N)(ρ_neighbours)) / (1 + 4*dt*diff*N*N)

Projection: 
    a = 1, c = 4
    ρ - (1/4)(ρ_neighbours) = div
    => ρ = (div + (1/4)(ρ_neighbours)) / 1

*/
void lin_solve(int b, std::vector<float>& x, const std::vector<float>& x0,
               float a, float c, int N) {
    for (int it = 0; it < 12; it++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                x[IX(i, j, N)] =
                    (x0[IX(i, j, N)] +
                     a * (x[IX(i - 1, j, N)] + x[IX(i + 1, j, N)] +
                          x[IX(i, j - 1, N)] + x[IX(i, j + 1, N)])) / c;
            }
        }
        set_bnd(b, x, N);
    }
}

static void solve_tridiagonal(const std::vector<float>& lower,
                              const std::vector<float>& diag,
                              const std::vector<float>& upper,
                              const std::vector<float>& rhs,
                              std::vector<float>& out,
                              int N) {
    std::vector<float> cprime(N + 1, 0.0f);
    std::vector<float> dprime(N + 1, 0.0f);

    cprime[1] = upper[1] / diag[1];
    dprime[1] = rhs[1] / diag[1];

    for (int i = 2; i <= N; i++) {
        float denom = diag[i] - lower[i] * cprime[i - 1];
        cprime[i] = (i < N) ? (upper[i] / denom) : 0.0f;
        dprime[i] = (rhs[i] - lower[i] * dprime[i - 1]) / denom;
    }

    out[N] = dprime[N];
    for (int i = N - 1; i >= 1; i--) {
        out[i] = dprime[i] - cprime[i] * out[i + 1];
    }
}

/*
lin_solve2: direct separable Helmholtz solver (FISHPAK-style equivalent)

Solves interior equation:
    c*x(i,j) - a*(x(i-1,j)+x(i+1,j)+x(i,j-1)+x(i,j+1)) = x0(i,j)

This is a direct solve for the interior grid and then applies set_bnd() boundary handling.
*/
void lin_solve2(int b, std::vector<float>& x, const std::vector<float>& x0,
                float a, float c, int N) {
    if (N <= 0) return;

    const float inv_norm = 2.0f / (N + 1.0f);
    const float pi = 3.14159265f;

    std::vector<float> sin_table((N + 1) * (N + 1), 0.0f);
    for (int k = 1; k <= N; k++) {
        for (int j = 1; j <= N; j++) {
            sin_table[k * (N + 1) + j] = std::sin(pi * k * j / (N + 1.0f));
        }
    }

    std::vector<float> rhs_hat((N + 2) * (N + 2), 0.0f);
    for (int i = 1; i <= N; i++) {
        for (int k = 1; k <= N; k++) {
            float acc = 0.0f;
            for (int j = 1; j <= N; j++) {
                acc += x0[IX(i, j, N)] * sin_table[k * (N + 1) + j];
            }
            rhs_hat[IX(i, k, N)] = acc;
        }
    }

    std::vector<float> u_hat((N + 2) * (N + 2), 0.0f);
    std::vector<float> lower(N + 1, 0.0f);
    std::vector<float> diag(N + 1, 0.0f);
    std::vector<float> upper(N + 1, 0.0f);
    std::vector<float> rhs_line(N + 1, 0.0f);
    std::vector<float> sol_line(N + 1, 0.0f);

    for (int k = 1; k <= N; k++) {
        float lambda_y = 2.0f * std::cos(pi * k / (N + 1.0f));
        float d = c - a * lambda_y;

        for (int i = 1; i <= N; i++) {
            lower[i] = -a;
            diag[i] = d;
            upper[i] = -a;
            rhs_line[i] = rhs_hat[IX(i, k, N)];
        }
        lower[1] = 0.0f;
        upper[N] = 0.0f;

        solve_tridiagonal(lower, diag, upper, rhs_line, sol_line, N);

        for (int i = 1; i <= N; i++) {
            u_hat[IX(i, k, N)] = sol_line[i];
        }
    }

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            float acc = 0.0f;
            for (int k = 1; k <= N; k++) {
                acc += u_hat[IX(i, k, N)] * sin_table[k * (N + 1) + j];
            }
            x[IX(i, j, N)] = inv_norm * acc;
        }
    }

    set_bnd(b, x, N);
}
