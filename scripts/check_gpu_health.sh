#!/bin/bash
#
# check_gpu_health.sh — Query GPU clocks, ECC, and throttle status via nvidia-smi
#
# Usage:
#   Local:   bash check_gpu_health.sh
#   Cluster: srun -N <nodes> --ntasks-per-node=1 -p batch bash check_gpu_health.sh
#
set -euo pipefail

HOSTNAME=$(hostname -s)

echo "=== ${HOSTNAME} ==="

# Clocks: current and max application clocks (graphics + memory)
echo ""
echo "-- Clocks --"
nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.max.memory \
    --format=csv,noheader,nounits \
| while IFS=', ' read -r idx gc gmax mc mmax; do
    lock=""
    if [ "$gc" != "$gmax" ]; then lock=" [NOT AT MAX]"; fi
    printf "  GPU %s: GFX %s/%s MHz  MEM %s/%s MHz%s\n" "$idx" "$gc" "$gmax" "$mc" "$mmax" "$lock"
done

# ECC status
echo ""
echo "-- ECC --"
nvidia-smi --query-gpu=index,ecc.mode.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader,nounits \
| while IFS=', ' read -r idx ecc_mode corr uncorr; do
    flag=""
    if [ "$ecc_mode" != "Enabled" ]; then flag=" [ECC DISABLED]"; fi
    if [ "$uncorr" != "0" ] && [ "$uncorr" != "N/A" ]; then flag+=" [UNCORRECTED ERRORS]"; fi
    printf "  GPU %s: ECC=%s  corrected=%s  uncorrected=%s%s\n" "$idx" "$ecc_mode" "$corr" "$uncorr" "$flag"
done

# Throttle reasons & temperature
echo ""
echo "-- Throttle & Temp --"
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown \
    --format=csv,noheader,nounits \
| while IFS=', ' read -r idx temp power hw_slow sw_therm; do
    flag=""
    if [ "$hw_slow" = "Active" ]; then flag=" [HW THROTTLE]"; fi
    if [ "$sw_therm" = "Active" ]; then flag+=" [THERMAL THROTTLE]"; fi
    printf "  GPU %s: %s°C  %sW%s\n" "$idx" "$temp" "$power" "$flag"
done
