#!/bin/bash

N_GPUS=$(nvidia-smi -L | wc -l)

if [[ "$N_GPUS" == "0" || "$N_GPUS" == "1" ]]; then
  python "$@"
else
  python -m torch.distributed.launch --nproc_per_node $N_GPUS "$@"
fi

