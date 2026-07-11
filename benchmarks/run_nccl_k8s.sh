#!/bin/bash
#
# run_nccl_k8s.sh — Submit an NCCL test job to Kubernetes
#
# Usage:
#   ./benchmarks/run_nccl_k8s.sh
#   ./benchmarks/run_nccl_k8s.sh -t all_gather -b 1M -B 8G
#   ./benchmarks/run_nccl_k8s.sh --build --import
#   ./benchmarks/run_nccl_k8s.sh --dry-run
#
# Available tests: all_reduce, all_gather, broadcast, reduce, reduce_scatter,
#                  alltoall, scatter, gather, sendrecv, hypercube
#
set -euo pipefail

BENCHMARKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$BENCHMARKS_DIR")"
TOOLS_DIR="${REPO_ROOT}/tools"

# ──────────────────── defaults ────────────────────
TEST="all_reduce"
GPUS_PER_NODE=8
MIN_BYTES="1M"
MAX_BYTES="8G"
STEP_FACTOR=2
ITERS=20
WARMUP=5
EXTRA_ARGS=""
IMAGE="togethercomputer/training-performance:nccl-tests-ub22.04-cuda12.9-v0.1"
NAMESPACE="default"
NODE_SELECTOR_KEY="node-role.together.ai/worker"
NODE_SELECTOR_VAL="true"
PULL_SECRET="dockerhub-nccl-tests"
TTL=3600
DO_BUILD=0
DO_IMPORT=0
DRY_RUN=0
JOB_NAME=""
OUTPUT_DIR=""
FOLLOW_LOGS=1

# ──────────────────── usage ────────────────────
usage() {
    cat <<'USAGE'
Usage: run_nccl_k8s.sh [OPTIONS]

Options:
  -t, --test NAME          Test name: all_reduce, all_gather, etc. (default: all_reduce)
  -g, --gpus NUM           GPUs (default: 8)
  -b, --min-bytes SIZE     Min message size (default: 1M)
  -B, --max-bytes SIZE     Max message size (default: 8G)
  -f, --factor NUM         Step factor (default: 2)
  -i, --iters NUM          Iterations per size (default: 20)
  -w, --warmup NUM         Warmup iterations (default: 5)
  -e, --extra ARGS         Extra args passed to *_perf binary
      --image IMAGE        Container image (default: docker.io/library/nccl-tests:cuda12.9)
      --namespace NS       Kubernetes namespace (default: default)
      --job-name NAME      Override job name (default: nccl-<test>-<timestamp>)
  -o, --output-dir DIR     Save logs to DIR after completion
      --build              Run tools/docker_build.sh before submitting
      --import             Run tools/containerd_import.sh before submitting
      --no-follow          Submit and exit without streaming logs
      --dry-run            Print generated YAML without submitting
  -h, --help               Show this help

Examples:
  # Quick run with defaults (all_reduce, 8 GPUs, 1M-8G)
  ./benchmarks/run_nccl_k8s.sh

  # Build image, import to containerd, then run all_gather
  ./benchmarks/run_nccl_k8s.sh --build --import -t all_gather

  # Dry-run to inspect generated YAML
  ./benchmarks/run_nccl_k8s.sh -t reduce_scatter --dry-run
USAGE
    exit 0
}

# ──────────────────── parse args ────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--test)        TEST="$2";           shift 2 ;;
        -g|--gpus)        GPUS_PER_NODE="$2";  shift 2 ;;
        -b|--min-bytes)   MIN_BYTES="$2";      shift 2 ;;
        -B|--max-bytes)   MAX_BYTES="$2";      shift 2 ;;
        -f|--factor)      STEP_FACTOR="$2";    shift 2 ;;
        -i|--iters)       ITERS="$2";          shift 2 ;;
        -w|--warmup)      WARMUP="$2";         shift 2 ;;
        -e|--extra)       EXTRA_ARGS="$2";     shift 2 ;;
        --image)          IMAGE="$2";          shift 2 ;;
        --namespace)      NAMESPACE="$2";      shift 2 ;;
        --job-name)       JOB_NAME="$2";       shift 2 ;;
        -o|--output-dir)  OUTPUT_DIR="$2";     shift 2 ;;
        --build)          DO_BUILD=1;          shift ;;
        --import)         DO_IMPORT=1;         shift ;;
        --no-follow)      FOLLOW_LOGS=0;       shift ;;
        --dry-run)        DRY_RUN=1;           shift ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ──────────────────── derived values ────────────────────
BINARY="${TEST}_perf"

if [[ -z "$JOB_NAME" ]]; then
    JOB_NAME="nccl-$(echo "$TEST" | tr '_' '-')-$(date +%Y%m%d-%H%M%S)"
fi

echo "======================================================"
echo "  NCCL Test — Kubernetes"
echo "======================================================"
echo "  Test:       ${BINARY}"
echo "  Sizes:      ${MIN_BYTES} -> ${MAX_BYTES} (factor ${STEP_FACTOR})"
echo "  Iters:      ${WARMUP} warmup + ${ITERS} measured"
echo "  GPUs:       ${GPUS_PER_NODE}"
echo "  Image:      ${IMAGE}"
echo "  Job name:   ${JOB_NAME}"
echo "  Namespace:  ${NAMESPACE}"
echo "======================================================"

# ──────────────────── optional pre-steps ────────────────────
if [[ $DO_BUILD -eq 1 ]]; then
    echo ""
    echo "[pre] Building Docker image..."
    bash "${TOOLS_DIR}/docker_build.sh"
fi

if [[ $DO_IMPORT -eq 1 ]]; then
    echo ""
    echo "[pre] Importing image into containerd..."
    bash "${TOOLS_DIR}/containerd_import.sh"
fi

# ──────────────────── generate job YAML ────────────────────
JOB_YAML=$(cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
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
            - ${BINARY}
            - "-b"
            - "${MIN_BYTES}"
            - "-e"
            - "${MAX_BYTES}"
            - "-f"
            - "${STEP_FACTOR}"
            - "-g"
            - "${GPUS_PER_NODE}"
            - "-n"
            - "${ITERS}"
            - "-w"
            - "${WARMUP}"$(
              if [[ -n "$EXTRA_ARGS" ]]; then
                  for arg in $EXTRA_ARGS; do echo ""; printf '            - "%s"' "$arg"; done
              fi
            )
          env:
            - name: NCCL_DEBUG
              value: "WARN"
          resources:
            limits:
              nvidia.com/gpu: "${GPUS_PER_NODE}"
          securityContext:
            capabilities:
              add: ["IPC_LOCK"]
EOF
)

# ──────────────────── dry-run ────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "--- DRY RUN: generated YAML ---"
    echo "$JOB_YAML"
    echo "--- end ---"
    exit 0
fi

# ──────────────────── submit ────────────────────
echo ""
echo "Submitting job..."
echo "$JOB_YAML" | kubectl apply -f -

# Wait for pod
echo ""
echo "Waiting for pod..."
POD=""
for i in $(seq 1 30); do
    POD=$(kubectl get pod -n "${NAMESPACE}" -l "job-name=${JOB_NAME}" \
          -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "$POD" ]] && break
    sleep 2
done

if [[ -z "${POD:-}" ]]; then
    echo "ERROR: Pod did not appear within 60s"
    kubectl describe job "${JOB_NAME}" -n "${NAMESPACE}"
    exit 1
fi

echo "Pod: ${POD}"

# Wait for running/completed
for i in $(seq 1 30); do
    PHASE=$(kubectl get pod "${POD}" -n "${NAMESPACE}" \
            -o jsonpath='{.status.phase}' 2>/dev/null || true)
    [[ "$PHASE" == "Running" || "$PHASE" == "Succeeded" || "$PHASE" == "Failed" ]] && break
    echo "  Waiting (phase: ${PHASE:-Pending})..."
    sleep 2
done

# ──────────────────── logs ────────────────────
if [[ $FOLLOW_LOGS -eq 1 ]]; then
    echo ""
    echo "--- Streaming logs ---"
    kubectl logs -f "${POD}" -n "${NAMESPACE}"
    echo "--- End of logs ---"
fi

# ──────────────────── save output ────────────────────
if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    OUT_FILE="${OUTPUT_DIR}/${JOB_NAME}.log"
    kubectl logs "${POD}" -n "${NAMESPACE}" > "${OUT_FILE}"
    echo ""
    echo "Logs saved to: ${OUT_FILE}"
fi

# ──────────────────── final status ────────────────────
echo ""
SUCCEEDED=$(kubectl get job "${JOB_NAME}" -n "${NAMESPACE}" \
            -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")
if [[ "$SUCCEEDED" == "1" ]]; then
    echo "Job SUCCEEDED: ${JOB_NAME}"
else
    echo "Job FAILED: ${JOB_NAME}"
    kubectl describe pod "${POD}" -n "${NAMESPACE}"
    exit 1
fi
