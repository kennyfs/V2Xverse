for i in $(seq 0 1); do
  port=$((40000 + i*2))
  CUDA_VISIBLE_DEVICES=$i ./external_paths/carla_root/CarlaUE4.sh \
    --world-port=$port -opengl -RenderOffScreen &
done
