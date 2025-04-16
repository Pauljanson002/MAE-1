#!/bin/bash

# Configuration
schedulers=("cosine")
reduction_factors=(0)
cooldown_ratios=(0.4)
constant_lr_ratios=(0.75)
alpha=(0.5 0.25 0.75)
lamda=(0.0)
seed=0
script="bin/batch.sh"
prefix="exp_6"
NUM_GPUS=8

# Function to generate all job combinations
generate_jobs() {
    local jobs=()
    for reduction_factor in "${reduction_factors[@]}"; do
        if [ $(echo "$reduction_factor == 0" | bc) -eq 1 ]; then
            methods=("lwf")
        else
            methods=("mas")
        fi
        for scheduler in "${schedulers[@]}"; do
            for method in "${methods[@]}"; do
                if [ "${scheduler}" = "cosine" ]; then
                    if [ "${method}" = "mas" ]; then
                        for a in "${alpha[@]}"; do
                            for l in "${lamda[@]}"; do
                                jobs+=("${scheduler}|${reduction_factor}|${method}|0|0|${a}|${l}")
                            done
                        done
                        continue
                    elif [ "${method}" = "lwf" ]; then
                        for a in "${alpha[@]}"; do
                            jobs+=("${scheduler}|${reduction_factor}|${method}|0|0|${a}|0")
                        done
                        continue
                    fi
                    jobs+=("${scheduler}|${reduction_factor}|${method}|0|0")
                    continue
                fi
                for cooldown_ratio in "${cooldown_ratios[@]}"; do
                    for constant_lr_ratio in "${constant_lr_ratios[@]}"; do
                        jobs+=("${scheduler}|${reduction_factor}|${method}|${cooldown_ratio}|${constant_lr_ratio}")
                    done
                done
            done
        done
    done
    echo "${jobs[@]}"
}

# Create tmux sessions and distribute jobs
distribute_jobs() {
    local jobs=($1)
    local num_jobs=${#jobs[@]}
    local jobs_per_gpu=$(( (num_jobs + NUM_GPUS - 1) / NUM_GPUS ))
    
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        local session_name="gpu${gpu}"
        
        # Create new tmux session
        tmux new-session -d -s "$session_name"
        
        # Set CUDA_VISIBLE_DEVICES for this session
        tmux send-keys -t "$session_name" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
        
        # Calculate job range for this GPU
        local start_idx=$((gpu * jobs_per_gpu))
        local end_idx=$(( (gpu + 1) * jobs_per_gpu ))
        if [ $end_idx -gt $num_jobs ]; then
            end_idx=$num_jobs
        fi
        
        # Create the job execution script for this GPU
        local gpu_script="gpu${gpu}_jobs.sh"
        echo "#!/bin/bash" > "$gpu_script"
        echo "export CUDA_VISIBLE_DEVICES=$gpu" >> "$gpu_script"
        
        # Add jobs to the script
        for ((i=start_idx; i<end_idx; i++)); do
            IFS='|' read -r scheduler reduction_factor method cooldown_ratio constant_lr_ratio alpha lamda <<< "${jobs[$i]}"
            echo "echo 'Starting job: scheduler=${scheduler}, reduction_factor=${reduction_factor}, method=${method}, cooldown_ratio=${cooldown_ratio}, constant_lr_ratio=${constant_lr_ratio}, alpha=${alpha}, lamda=${lamda}'" >> "$gpu_script"
            echo "$script \"${scheduler}\" \"${reduction_factor}\" \"${method}\" \"${seed}\" \"${prefix}\" \"${cooldown_ratio}\" \"${constant_lr_ratio}\" \"${alpha}\" \"${lamda}\"" >> "$gpu_script"
            echo "sleep 1" >> "$gpu_script"
        done
        
        chmod +x "$gpu_script"
        
        # Start executing the script in the tmux session
        #tmux send-keys -t "$session_name" "./$gpu_script" C-m
    done
}

# Main execution
echo "Generating job combinations..."
jobs=$(generate_jobs)
echo "Distributing jobs across GPUs..."
distribute_jobs "$jobs"

echo "All jobs have been distributed across tmux sessions."
echo "To attach to a specific session, use: tmux attach-session -t gpu<N>"
echo "To list all sessions, use: tmux ls"