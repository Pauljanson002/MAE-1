#!/bin/bash

# Define parameters
schedulers=("cosine" "infinite_cosine")
reduction_factors=(0 0.1 0.2 0.4 0.5)
cooldown_ratios=(0.4 0.5 0.6 0.7)
constant_lr_ratios=(0.25 0.5 0.6 0.75)
seed=2222
script=bin/speed/batch.sh
prefix="exp"

# Define GPU MIG IDs
mig_ids=(
    "MIG-ac63ec32-7bf5-5415-b1db-03ea60c58446"
    "MIG-d210cb3f-9820-5f5b-9691-21822446d88c"
    "MIG-c680c5b7-2cc7-5e7d-92e3-a36c69dafb9b"
    "MIG-2773c012-1196-5f82-9055-8392b5006ff6"
)
# Create 4 tmux sessions
for i in {0..3}; do
    tmux new-session -d -s "$i"
    tmux send-keys -t "$i" "export CUDA_VISIBLE_DEVICES=${mig_ids[$i]}" C-m
done

counter=0
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
                tmux_session=$((counter % 4))
                tmux send-keys -t "$tmux_session" "${script} ${scheduler} ${reduction_factor} ${method} ${seed} ${prefix} 0 0 ; wait" C-m
                sleep 1
                counter=$((counter+1))
                continue
            fi
            for cooldown_ratio in "${cooldown_ratios[@]}"; do
                for constant_lr_ratio in "${constant_lr_ratios[@]}"; do
                    echo "Submitting job: scheduler=${scheduler}, reduction_factor=${reduction_factor}, method=${method}, seed=${seed}, cooldown_ratio=${cooldown_ratio}, constant_lr_ratio=${constant_lr_ratio}"
                    name=${scheduler:0:3}_${reduction_factor}_${method}_${cooldown_ratio}
                    tmux_session=$((counter % 4))
                    tmux send-keys -t "$tmux_session" "${script} ${scheduler} ${reduction_factor} ${method} ${seed} ${prefix} ${cooldown_ratio} ${constant_lr_ratio} ; wait" C-m
                    # Add small delay to avoid overwhelming the scheduler
                    sleep 1
                    counter=$((counter+1))
                done
            done
        done
    done
done

