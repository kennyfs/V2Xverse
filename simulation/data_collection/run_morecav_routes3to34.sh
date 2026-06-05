#!/bin/bash
# Generate weather-0-morecav dataset for routes 3-34 (EGO_NUM=5)
# Routes 3-4  → test split supplement
# Routes 5-34 → training split
# Run from V2Xverse root directory.

CARLA_ROOT=external_paths/carla_root
DATA_ROOT=external_paths/data_root

mkdir -p ${DATA_ROOT}/weather-0-morecav/data
mkdir -p ${DATA_ROOT}/weather-0-morecav/results

ROUTES=()
for i in $(seq 3 34); do
    ROUTES+=("simulation/data_collection/scripts/weather-0-morecav/routes_town05_${i}.sh")
done

for ROUTE_SCRIPT in "${ROUTES[@]}"; do
    ROUTE_NAME=$(basename ${ROUTE_SCRIPT} .sh)
    echo "========================================"
    echo " Starting CARLA for route: ${ROUTE_NAME}"
    echo "========================================"

    # Start CARLA server
    ${CARLA_ROOT}/CarlaUE4.sh \
        --world-port=40000 \
        -opengl \
        -RenderOffScreen &
    CARLA_PID=$!
    echo "CARLA PID: ${CARLA_PID}"

    # Poll until CARLA port 40000 is ready (max 120s)
    echo "Waiting for CARLA on port 40000..."
    for i in $(seq 1 24); do
        sleep 5
        if nc -z localhost 40000 2>/dev/null; then
            echo "  CARLA up after $((i*5))s"
            break
        fi
        echo "  ...waiting (${i}/24)"
    done
    sleep 10

    echo "Running: ${ROUTE_NAME}"
    bash ${ROUTE_SCRIPT}
    ROUTE_STATUS=$?

    echo "Route done (exit ${ROUTE_STATUS}). Killing CARLA..."
    kill ${CARLA_PID} 2>/dev/null || true
    pkill -9 -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true

    # Wait for ports to be released
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
        echo "WARNING: ${ROUTE_NAME} exited with code ${ROUTE_STATUS}"
    fi
done

echo "========================================"
echo " All routes done. Rebuilding dataset index..."
echo "========================================"
python simulation/data_collection/gen_index_morecav.py

echo "Done."
