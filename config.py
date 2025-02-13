import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--finetune_batch_size", type=int, default=128)
    parser.add_argument("--max_device_batch_size", type=int, default=1024)
    parser.add_argument("--base_learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--finetune_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--total_epoch", type=int, default=400)
    parser.add_argument("--finetune_epoch", type=int, default=100)
    parser.add_argument("--finetune_warmup_epoch", type=int, default=5)
    parser.add_argument("--warmup_epoch", type=int, default=20)
    # parser.add_argument("--model_path", type=str, default="vit-t-mae.pt")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--name", type=str, default="first")
    parser.add_argument("--reduction_factor", type=float, default=0.05, help="Reduction factor buffer reduction")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--cooldown_ratio", type=float, default=0.3, help="Cooldown ratio")
    parser.add_argument("--constant_lr_ratio", type=float, default=0.25, help="Constant lr ratio")
    parser.add_argument("--constant_ratio", type=float, default=0.8, help="Constant ratio")
    
    
    return parser.parse_args()
