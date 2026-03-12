#!/usr/bin/env bash
#SBATCH --job-name=fluids-bench
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# This code is OpenMP-only. The multi-node runs in this script measure
# throughput by launching one independent simulation per node, not
# distributed time-to-solution scaling of a single simulation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_COUNTS=(1 2 4 8)

GRID_SIZE="${GRID_SIZE:-256}"
STEPS="${STEPS:-200}"
THREADS_PER_RANK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}"

submit_jobs() {
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch is required to submit the benchmark jobs." >&2
        exit 1
    fi

    for nodes in "${NODE_COUNTS[@]}"; do
        sbatch \
            --nodes="${nodes}" \
            --job-name="fluids-n${nodes}" \
            --export=ALL,BENCH_NODES="${nodes}",GRID_SIZE="${GRID_SIZE}",STEPS="${STEPS}",CPUS_PER_TASK="${THREADS_PER_RANK}" \
            "$0"
    done
}

write_summary() {
    local out_dir="$1"
    local nodes="$2"
    local summary_file="${out_dir}/summary.txt"

    awk -F= -v nodes="${nodes}" -v threads="${THREADS_PER_RANK}" '
        /^elapsed=/ {
            value = $2 + 0.0;
            sum += value;
            count += 1;
            if (count == 1 || value < min) min = value;
            if (value > max) max = value;
        }
        END {
            if (count == 0) {
                exit 1;
            }
            printf "nodes=%s\nthreads_per_rank=%s\nruns=%d\navg_elapsed=%.6f\nmin_elapsed=%.6f\nmax_elapsed=%.6f\n",
                   nodes, threads, count, sum / count, min, max;
        }
    ' "${out_dir}"/rank_*/time.log > "${summary_file}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit_jobs
    exit 0
fi

cd "${ROOT_DIR}"
make

NODES="${BENCH_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
OUT_DIR="${ROOT_DIR}/build/benchmarks/nodes_${NODES}_job_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

cat > "${OUT_DIR}/metadata.txt" <<EOF
grid_size=${GRID_SIZE}
steps=${STEPS}
nodes=${NODES}
threads_per_rank=${THREADS_PER_RANK}
omp_proc_bind=${OMP_PROC_BIND}
omp_places=${OMP_PLACES}
job_id=${SLURM_JOB_ID}
EOF

srun \
    --nodes="${NODES}" \
    --ntasks="${NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${THREADS_PER_RANK}" \
    --kill-on-bad-exit=1 \
    bash -lc '
        set -euo pipefail
        repo_root="'"${ROOT_DIR}"'"
        out_dir="'"${OUT_DIR}"'"
        rank="${SLURM_PROCID}"
        run_dir="${out_dir}/rank_${rank}"
        mkdir -p "${run_dir}/build"
        cd "${run_dir}"
        export OMP_NUM_THREADS="'"${THREADS_PER_RANK}"'"
        export OMP_PROC_BIND="'"${OMP_PROC_BIND}"'"
        export OMP_PLACES="'"${OMP_PLACES}"'"
        /usr/bin/time -f "elapsed=%e" "${repo_root}/build/main" "'"${GRID_SIZE}"'" "'"${STEPS}"'" > stdout.log 2> time.log
    '

write_summary "${OUT_DIR}" "${NODES}"
cat "${OUT_DIR}/summary.txt"