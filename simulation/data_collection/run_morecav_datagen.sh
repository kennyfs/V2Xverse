#!/bin/bash
# Generate weather-0-morecav dataset (EGO_NUM=6, pedestrian_amount=200)
# Runs one CARLA instance and three Town05 routes sequentially.
# Usage: bash simulation/data_collection/run_morecav_datagen.sh
# Run from V2Xverse root directory.

CARLA_ROOT=external_paths/carla_root
DATA_ROOT=external_paths/data_root

mkdir -p ${DATA_ROOT}/weather-0-morecav/data
mkdir -p ${DATA_ROOT}/weather-0-morecav/results

ROUTES=(
    "simulation/data_collection/scripts/weather-0-morecav/routes_town05_0.sh"
    "simulation/data_collection/scripts/weather-0-morecav/routes_town05_1.sh"
    "simulation/data_collection/scripts/weather-0-morecav/routes_town05_2.sh"
)

for ROUTE_SCRIPT in "${ROUTES[@]}"; do
    ROUTE_NAME=$(basename ${ROUTE_SCRIPT} .sh)
    echo "========================================"
    echo " Starting CARLA for route: ${ROUTE_NAME}"
    echo "========================================"

    # Start CARLA server with offscreen OpenGL rendering
    CUDA_VISIBLE_DEVICES=0 ${CARLA_ROOT}/CarlaUE4.sh \
        --world-port=40000 \
        -opengl \
        -RenderOffScreen &
    CARLA_PID=$!
    echo "CARLA PID: ${CARLA_PID}"

    # Poll until CARLA port 40000 is accepting connections (max 120s)
    echo "Waiting for CARLA to be ready on port 40000..."
    for i in $(seq 1 24); do
        sleep 5
        if nc -z localhost 40000 2>/dev/null; then
            echo "CARLA is up after $((i*5))s"
            break
        fi
        echo "  ...still waiting (${i}/24)"
    done
    # Extra settle time after port opens
    sleep 10

    echo "Running route: ${ROUTE_NAME}"
    bash ${ROUTE_SCRIPT}
    ROUTE_STATUS=$?

    echo "Route finished (exit code ${ROUTE_STATUS}). Killing CARLA..."
    kill ${CARLA_PID} 2>/dev/null || true
    pkill -9 -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
    # Wait for both CARLA ports to be fully released before next route
    echo "Waiting for ports 40000 and 40500 to be released..."
    for i in $(seq 1 30); do
        sleep 3
        if ! ss -tlnp 2>/dev/null | grep -qE ':40000|:40500'; then
            echo "  Ports released after $((i*3))s"
            break
        fi
        echo "  ...ports still bound (${i}/30)"
    done
    sleep 5

    if [ ${ROUTE_STATUS} -ne 0 ]; then
        echo "WARNING: route ${ROUTE_NAME} exited with code ${ROUTE_STATUS}"
    fi
done

echo "========================================"
echo " All routes done. Rebuilding dataset index..."
echo "========================================"
python simulation/data_collection/gen_index_morecav.py

echo "Done. Data saved to: ${DATA_ROOT}/weather-0-morecav/data"
