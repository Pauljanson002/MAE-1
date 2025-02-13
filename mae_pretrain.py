import math
import torch
from model import MAE_ViT, ViT_Classifier
from utils import setup_seed
from config import get_args
from data import get_pretrain_dataloader,get_finetune_dataloader
from trainer import MAETrainer
from logger import Logger
torch.set_float32_matmul_precision("high")


def main():
    args = get_args()
    args.lr = args.base_learning_rate * args.batch_size / 256
    setup_seed(args.seed)
    logger = Logger()
    logger.init(
        "mae_baselines",
        args.name,
        args.output_dir,
        config=args,
    )

    model = MAE_ViT(mask_ratio=args.mask_ratio).to("cuda")
    trainer = MAETrainer(model, args)

    for task_id in range(args.num_tasks):
        # Pretrain
        if task_id > 0:
            trainer.reset_optimizer()
            if args.scheduler != "cosine":
                trainer.load_annealed_model(task_id=task_id - 1)
        trainer.unfreeze_model()
        logger.print(f"Task {task_id}")
        dataloader = get_pretrain_dataloader(task_id, args)
        steps_per_update = args.batch_size // args.max_device_batch_size
        for epoch in range(args.total_epoch):
            trainer.train_epoch(dataloader, epoch, steps_per_update)
            if epoch % 50 == 0 or epoch == args.total_epoch - 1:
                trainer.save_model(epoch=epoch,task_id=task_id)

            if args.scheduler != "cosine" and epoch == math.floor(args.total_epoch * args.constant_ratio):
                trainer.save_annealed_model(task_id=task_id)
        
        trainer.freeze_model()
        finetune_dataloader,eval_dataloader = get_finetune_dataloader(task_id, args)
        finetune_model = ViT_Classifier(model.encoder,num_classes=2 * (task_id + 1)).to("cuda")
        finetune_steps_per_update = 1
        trainer.set_finetune_model(finetune_model)
        for epoch in range(args.finetune_epoch):
            trainer.finetune_epoch(finetune_dataloader, epoch, finetune_steps_per_update)
        
        trainer.start_evaluation(eval_dataloader,task_id)
        

        trainer.task_id += 1

    logger.finish()


if __name__ == "__main__":
    main()
