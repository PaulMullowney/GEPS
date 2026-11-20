#!/bin/bash
# mpiexec wrapper: each rank sees only its own GPU, entry mod(local rank, n) of the
# ROCR_VISIBLE_DEVICES list (the device device_init would pick with every GPU visible).
# With all GPUs visible every process also allocates on device 0, and rank 0's kernels
# ran 2x slower on TCo383 8-GPU (diabat 403 vs 196 ms/step).
# GEPS_RANK_GPU_MASK=0 leaves the list unchanged.
if [ "${GEPS_RANK_GPU_MASK:-1}" != "0" ] && [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
  r=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${MPI_LOCALRANKID:-0}}}
  IFS=, read -ra _devs <<< "${ROCR_VISIBLE_DEVICES}"
  if [ ${#_devs[@]} -gt 1 ]; then
    export ROCR_VISIBLE_DEVICES=${_devs[$((r % ${#_devs[@]}))]}
    export HIP_VISIBLE_DEVICES=0 GPU_DEVICE_ORDINAL=0
  fi
fi
exec "$@"
