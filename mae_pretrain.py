import math
import torch
from model import MAE_ViT, ViT_Classifier
from utils import setup_seed
from config import get_args
from data import get_pretrain_dataloader,get_finetune_dataloader
from trainer import MAETrainer
from mas_trainer import MASTrainer
from lwf_trainer import LWFTrainer
from gdumb_trainer import GDUMBTrainer
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

    if args.method == "mas":
        trainer = MASTrainer(model, args)
    elif args.method == "lwf":
        trainer = LWFTrainer(model, args)
    elif args.method == "gdumb":
        trainer = GDUMBTrainer(model, args)
    else:
        trainer = MAETrainer(model, args)

    for task_id in range(args.num_tasks):
        # Pretrain
        if task_id > 0:
            if args.scheduler != "cosine" and args.method != "gdumb":
                trainer.load_annealed_model(task_id=task_id - 1)
        trainer.unfreeze_model()
        logger.print(f"Task {task_id}")
        dataloader = get_pretrain_dataloader(task_id, args)
        steps_per_update = args.batch_size // args.max_device_batch_size
        for epoch in range(args.total_epoch):
            trainer.train_epoch(dataloader, epoch, steps_per_update)
            if epoch % 50 == 0 or epoch == args.total_epoch - 1:
                trainer.save_model(epoch=epoch,task_id=task_id)


        
        if args.method in ["mas","lwf"]:
            trainer.after_training_exp(task_id,dataloader)
        
        trainer.freeze_model()
        finetune_dataloader,eval_dataloader = get_finetune_dataloader(task_id, args)
        finetune_model = ViT_Classifier(model.encoder,num_classes=2 * (task_id + 1)).to("cuda")
        finetune_steps_per_update = 1
        trainer.set_finetune_model(finetune_model)
        for epoch in range(args.finetune_epoch):
            trainer.finetune_epoch(finetune_dataloader, epoch, finetune_steps_per_update)
        
        trainer.start_evaluation(eval_dataloader,task_id)
        
        if args.method == "gdumb":
            trainer.after_finetuning()

        trainer.task_id += 1
        
        if args.hpo == 1 and task_id == 1:
            logger.print("HPO mode, exiting after two tasks")
            break

    logger.finish()
    



if __name__ == "__main__":
    main()
