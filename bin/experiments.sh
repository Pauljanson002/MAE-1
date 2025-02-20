#!/bin/bash

# schedulers=("cosine" "infinite_cosine")
# reduction_factors=(0 0.05 0.1 0.2)
schedulers=("cosine")
reduction_factors=(0.4)
seed=0
script=bin/batch.sh
runs=1


for reduction_factor in "${reduction_factors[@]}"; do
    if [ $(echo "$reduction_factor == 0" | bc) -eq 1 ]; then
        # When reduction_factor is 0, run all methods including lwf
        # methods=("base" "mas" "lwf")
        methods=("base")
    else
        # When reduction_factor > 0, run base, mas, gdumb
        methods=("base")
    fi

    for scheduler in "${schedulers[@]}"; do
        for method in "${methods[@]}"; do
            echo "Submitting job: scheduler=${scheduler}, reduction_factor=${reduction_factor}, method=${method} seed=${seed}"
            name=${scheduler:0:3}_${reduction_factor}_${method}
            $script "${scheduler}" "${reduction_factor}" "${method}" "${seed}" "try_1"
            # Add small delay to avoid overwhelming the scheduler
            sleep 1
        done
    done
done

