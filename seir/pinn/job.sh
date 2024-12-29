#!/bin/bash

# Maximum number of parallel jobs
MAX_JOBS=30

# Create a semaphore function to manage parallel jobs
function semaphore {
  local N=$1
  while (( $(jobs | wc -l) >= N )); do
    sleep 1
    jobs > /dev/null
  done
}

# Loop through all the seeds and run the jobs
for (( seed=399; seed>=0; seed-- )); do
  # Use the semaphore to control the parallel jobs
  semaphore $MAX_JOBS
  echo "Running script with seed $seed"
  PYTHONPATH=. python pinn/seir_v2_pinn_main.py $seed &
done

# Wait for all background jobs to complete
wait
