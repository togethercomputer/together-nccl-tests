#!/bin/bash
#
# run_k8s.sh — NCCL benchmark suite via Kubernetes Jobs
#
# Submits one K8s Job per test config, streams logs, saves results locally.
# GPU type is auto-detected (or set with --gpu-type) and determines the
# results subdirectory: benchmarks/<GPU_TYPE>/results/<timestamp>_k8s/
#
# Usage:
#   ./benchmarks/run_k8s.sh
#   ./benchmarks/run_k8s.sh --gpu-type B200 --min 1M --max 8G
#   ./benchmarks/run_k8s.sh --dry-run
#
set -euo pipefail

BENCHMARKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCHMARKS_DIR}/lib/common.sh"

# ── defaults ──────────────────────────────────────────────────────────────────
GPU_TYPE=""
GPUS=8
MIN_BYTES="1M"
MAX_BYTES="8G"
STEP_FACTOR=2
ITERS=20
WARMUP=5
IMAGE="togethercomputer/training-performance:nccl-tests-ub22.04-cuda12.9-v0.1"
NAMESPACE="default"
NODE_SELECTOR_KEY="node-role.together.ai/worker"
NODE_SELECTOR_VAL="true"
PULL_SECRET="dockerhub-nccl-tests"
TTL=3600
DRY_RUN=0
BASELINE_DIR=""
SET_BASELINE=0

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: run_k8s.sh [OPTIONS]

Options:
      --gpu-type TYPE       GPU type label for results dir (default: auto-detect)
  -b, --min SIZE            Min message size (default: 1M)
  -B, --max SIZE            Max message size (default: 8G)
  -f, --factor NUM          Step factor      (default: 2)
  -i, --iters NUM           Iterations       (default: 20)
  -w, --warmup NUM          Warmup iters     (default: 5)
      --image IMAGE         Container image
      --namespace NS        Kubernetes namespace (default: default)
      --pull-secret NAME    imagePullSecret name (default: dockerhub-nccl-tests)
      --ttl SECS            Job TTL after completion (default: 3600)
      --baseline DIR        Baseline dir for comparison
      --set-baseline        Mark this run as new baseline
      --dry-run             Print YAMLs without submitting
  -h, --help                Show this help
USAGE
    exit 0
}

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-type)     GPU_TYPE="$2";     shift 2 ;;
        -b|--min)       MIN_BYTES="$2";    shift 2 ;;
        -B|--max)       MAX_BYTES="$2";    shift 2 ;;
        -f|--factor)    STEP_FACTOR="$2";  shift 2 ;;
        -i|--iters)     ITERS="$2";        shift 2 ;;
        -w|--warmup)    WARMUP="$2";       shift 2 ;;
        --image)        IMAGE="$2";        shift 2 ;;
        --namespace)    NAMESPACE="$2";    shift 2 ;;
        --pull-secret)  PULL_SECRET="$2";  shift 2 ;;
        --ttl)          TTL="$2";          shift 2 ;;
        --baseline)     BASELINE_DIR="$2"; shift 2 ;;
        --set-baseline) SET_BASELINE=1;    shift ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── GPU type & results dir ────────────────────────────────────────────────────
[[ -z "$GPU_TYPE" ]] && GPU_TYPE=$(detect_gpu_type)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${BENCHMARKS_DIR}/${GPU_TYPE}/results/${TIMESTAMP}_k8s"
[[ -z "$BASELINE_DIR" ]] && BASELINE_DIR="${BENCHMARKS_DIR}/${GPU_TYPE}/results/baseline_k8s"
mkdir -p "$RESULTS_DIR"

echo "======================================================"
echo "  NCCL Benchmark Suite — Kubernetes"
echo "======================================================"
echo "  GPU:       ${GPU_TYPE}"
echo "  Sizes:     ${MIN_BYTES} -> ${MAX_BYTES} (factor ${STEP_FACTOR})"
echo "  Iters:     ${WARMUP} warmup + ${ITERS} measured"
echo "  Image:     ${IMAGE}"
echo "  Namespace: ${NAMESPACE}"
echo "  Results:   ${RESULTS_DIR}"
echo "======================================================"

# Sync imagePullSecret from DOCKER_USER/TOKEN if provided
registry_auth_k8s "$NAMESPACE" "$PULL_SECRET"

TOTAL=${#NCCL_TEST_MATRIX[@]}
PASSED=0
FAILED=0

# ── run one test ──────────────────────────────────────────────────────────────
run_test() {
    local test_name="$1" label="$2" nccl_algo="$3" nvls_enable="$4" collnet_enable="$5"
    local outfile="${RESULTS_DIR}/${test_name}_${label}.out"
    local binary="${test_name}_perf"
    local job_name="nccl-$(echo "${test_name}-${label}" | tr '_' '-')-$(date +%H%M%S)"

    echo ""
    echo "--- ${test_name} | config=${label} ---"

    local env_block=""
    [[ -n "$nccl_algo" ]] && env_block+="
            - name: NCCL_ALGO
              value: \"${nccl_algo}\""
    env_block+="
            - name: NCCL_NVLS_ENABLE
              value: \"${nvls_enable}\"
            - name: NCCL_COLLNET_ENABLE
              value: \"${collnet_enable}\"
            - name: NCCL_DEBUG
              value: \"WARN\""

    local job_yaml
    job_yaml=$(cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: ${NAMESPACE}
spec:
  ttlSecondsAfterFinished: ${TTL}
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        ${NODE_SELECTOR_KEY}: "${NODE_SELECTOR_VAL}"
      imagePullSecrets:
        - name: ${PULL_SECRET}
      containers:
        - name: nccl-test
          image: ${IMAGE}
          imagePullPolicy: IfNotPresent
          command:
            - ${binary}
            - "-b"
            - "${MIN_BYTES}"
            - "-e"
            - "${MAX_BYTES}"
            - "-f"
            - "${STEP_FACTOR}"
            - "-g"
            - "${GPUS}"
            - "-n"
            - "${ITERS}"
            - "-w"
            - "${WARMUP}"
          env:${env_block}
          resources:
            limits:
              nvidia.com/gpu: "${GPUS}"
          securityContext:
            capabilities:
              add: ["IPC_LOCK"]
EOF
)

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "$job_yaml"
        echo "(dry-run: would save to ${outfile})"
        return
    fi

    echo "$job_yaml" | kubectl apply -f - >/dev/null

    local pod=""
    for i in $(seq 1 30); do
        pod=$(kubectl get pod -n "${NAMESPACE}" -l "job-name=${job_name}" \
              -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
        [[ -n "$pod" ]] && break
        sleep 2
    done

    if [[ -z "${pod:-}" ]]; then
        echo "  ERROR: Pod did not appear"
        FAILED=$((FAILED + 1)); return 1
    fi

    for i in $(seq 1 60); do
        local phase
        phase=$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
                -o jsonpath='{.status.phase}' 2>/dev/null || true)
        [[ "$phase" == "Running" || "$phase" == "Succeeded" || "$phase" == "Failed" ]] && break
        sleep 2
    done

    kubectl logs -f "${pod}" -n "${NAMESPACE}" 2>/dev/null | tee "${outfile}"

    if grep -q "Out of bounds values : 0 OK" "${outfile}" 2>/dev/null; then
        echo "  => Peak busBW: $(peak_busbw "${outfile}") GB/s"
        PASSED=$((PASSED + 1))
    else
        echo "  ERROR: Job failed or produced no valid output"
        FAILED=$((FAILED + 1))
    fi
}

# ── run all tests ─────────────────────────────────────────────────────────────
IDX=0
for entry in "${NCCL_TEST_MATRIX[@]}"; do
    IDX=$((IDX + 1))
    IFS=':' read -r test_name label nccl_algo nvls_enable collnet_enable <<< "$entry"
    echo ""
    echo "[${IDX}/${TOTAL}] ${test_name} / ${label}"
    run_test "$test_name" "$label" "$nccl_algo" "$nvls_enable" "$collnet_enable" || true
done

# ── summary ───────────────────────────────────────────────────────────────────
[[ $DRY_RUN -eq 0 ]] && print_summary "${RESULTS_DIR}" "${BASELINE_DIR}"

echo ""
echo "Passed: ${PASSED} / ${TOTAL}  |  Failed: ${FAILED}"
echo "Results: ${RESULTS_DIR}"
[[ -d "${BASELINE_DIR}" ]] && echo "Baseline: ${BASELINE_DIR}"

if [[ $DRY_RUN -eq 0 && $SET_BASELINE -eq 1 ]]; then
    ln -sfn "$(basename "${RESULTS_DIR}")" \
        "${BENCHMARKS_DIR}/${GPU_TYPE}/results/baseline_k8s"
    echo "Baseline updated -> ${RESULTS_DIR}"
fi
