#!/bin/bash

# Define parameters
schedulers=("cosine" "infinite_cosine")
reduction_factors=(0 0.1 0.2 0.4 0.5)
cooldown_ratios=(0.4 0.5 0.6 0.7)
constant_lr_ratios=(0.25 0.5 0.6 0.75)
seed=42
script=bin/narval/batch.sh
prefix="exp"


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
            if [ "${scheduler}" = "cosine" ]; then
                echo "Submitting job: scheduler=${scheduler}, reduction_factor=${reduction_factor}, method=${method}, seed=${seed}"
                name=${scheduler:0:3}_${reduction_factor}_${method}
                sbatch --job-name=$name $script "${scheduler}" "${reduction_factor}" "${method}" "${seed}" ${prefix} "0" "0"
                sleep 1
                continue
            fi
            for cooldown_ratio in "${cooldown_ratios[@]}"; do
                for constant_lr_ratio in "${constant_lr_ratios[@]}"; do
                    echo "Submitting job: scheduler=${scheduler}, reduction_factor=${reduction_factor}, method=${method}, seed=${seed}, cooldown_ratio=${cooldown_ratio}, constant_lr_ratio=${constant_lr_ratio}"
                    name=${scheduler:0:3}_${reduction_factor}_${method}_${cooldown_ratio}
                    sbatch --job-name=$name $script "${scheduler}" "${reduction_factor}" "${method}" "${seed}" ${prefix} "${cooldown_ratio}" "${constant_lr_ratio}"
                    # Add small delay to avoid overwhelming the scheduler
                    sleep 1
                done
            done
        done
    done
done

