# Parallel CPU Fluid Simulation (C++ + OpenMP)

This project is a simple 2D fluid simulation written in lean C++ with OpenMP parallelisation on the CPU.

## Motivation

This project was built to:

1. Learn C++ by implementing a full fluid simulation from scratch.
2. Learn CPU parallelisation with OpenMP (`#pragma omp parallel for`).
3. Prepare for the AMD GPU Hackathon, where we will work with AMD and DiRAC to port this solver to GPU.

The longer-term goal is to benchmark equivalent solver versions across:

1. BlueBEAR (CPU)
2. Baskerville (GPU)
3. AMD Developer Cloud (GPU)

## What The Code Does

The simulation loosely follows `Stable Fluids, Stam` and evolves two coupled fields on a 2D grid:

1. Velocity field: $(u, v)$
2. Scalar density field: $\rho$ (smoke/dye)

Each time step:

1. Inject density and upward velocity at a source region.
2. Update velocity with diffusion, incompressibility projection, and advection.
3. Update density with diffusion, advection, and optional dissipation.
4. Write frames to `build/density.bin`.
5. Render frames to a GIF with `src/render.py`.

## Build And Run

Requirements:

1. `g++` with OpenMP support
2. `make`
3. Python 3 with `numpy` and `matplotlib`

Build:

```bash
make
```

Run simulation + render:

```bash
make run ARGS="128 100 1e-6 1e-6 2e-2"
```

`ARGS` map to:

1. `N` (grid size)
2. `steps`
3. `visc` (kinematic viscosity)
4. `diff` (density diffusivity)
5. `diss` (dissipation)
6. `lin_solve_iters` (optional, default `24`)
7. `solver_id` (optional: `0=lin_solve0`, `1=lin_solve1`, `3=lin_solve3`, `4=lin_solve4-experimental`; default `3`)

Performance build profiles:

```bash
make native
make release
make debug
```

MPI build and run (throughput scaling via one simulation per rank):

```bash
make mpi
make sim-mpi NP=4 ARGS="1024 200 1e-6 1e-6 2e-2 24 4"
```

GPU offload build and run (OpenMP target):

```bash
make gpu
make sim-gpu ARGS="2048 80 1e-6 1e-6 2e-2 24 4"
```

If your toolchain supports a specific GPU target, pass offload flags, for example:

```bash
make gpu GPU_CXX=clang++ GPU_OFFLOAD_FLAGS="-fopenmp-targets=nvptx64-nvidia-cuda"
```

OpenMP runtime knobs (used by `make run` and `make sim`):

```bash
make run OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ARGS="512 200 1e-6 1e-6 2e-2 24"
```

Render only from existing binary output:

```bash
make render
```

Output files:

1. Raw frames: `build/density.bin`
2. GIF: `sims/recent.gif`

## Solver Steps

### 1) Diffusion

Diffusion solves:

$$ \frac{d\phi}{dt} = \nu \cdot \nabla^2(\phi)$$

for either velocity components or density. After implicit time discretisation, this becomes a linear system solved by `lin_solve0` (Gauss-Siedel style), or `lin_solve1` (Jacobi). In code, this appears as:

`x = (x0 + a * neighbor_sum) / c`

with:

`a = dt * coeff * N^2`

where `coeff` is `visc` for velocity or `diff` for density.

### 2) Projection (Enforce Incompressibility)

For incompressible flow we need:

$\nabla \cdot u = 0$

Steps:

1. Compute discrete divergence from `(u, v)`.
2. Solve Poisson equation for pressure:

$\nabla^2 p = \nabla \cdot u$

3. Subtract pressure gradient:

$u \leftarrow u - \nabla p$

This removes divergent components and keeps the velocity field mass-conserving.

### 3) Advection (Semi-Lagrangian)

To advect field $\phi$:

$\phi(x, t + dt) = \phi(x - dt \cdot u(x, t), t)$

So each cell traces *backwards* along the velocity and samples the previous field. We use bilinear interpolation.

### 4) Dissipation

Density can be damped each step as:

$\rho \leftarrow \frac{\rho}{(1 + a \cdot dt)}$

where $a$ is the dissipation coefficient.

## Next Steps

1. Preserve solver correctness while porting kernels to GPU.
2. Keep the same inputs and output format for fair benchmarking.
3. Compare runtime and scaling across BlueBEAR, Baskerville, and AMD Developer Cloud.