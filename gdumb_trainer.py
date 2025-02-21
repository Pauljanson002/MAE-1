import math
import torch
from tqdm import tqdm
from model import MAE_ViT
from trainer import MAETrainer
from logger import Logger
import copy
import os
logger = Logger()


class GDUMBTrainer(MAETrainer):

    def __init__(self, model, args):
        super().__init__(model, args)

    def after_finetuning(self):
        self.model = MAE_ViT(mask_ratio=self.args.mask_ratio).to("cuda")

    def train_epoch(self, dataloader, epoch, steps_per_update):
        self.model.train()
        losses = []
        step_count = 0
        self.optim.zero_grad()
        sample_factor = 1 if self.task_id == 0 or self.args.reduction_factor == 0 else 2
        pbar = tqdm(dataloader, total=len(dataloader) * sample_factor)

        for data_iter, (img, _) in enumerate(pbar):
            step_count += 1

            predicted_img, mask = self.model(img)
            loss = (
                torch.mean((predicted_img - img) ** 2 * mask) / self.args.mask_ratio
            )

            loss.backward()

            if step_count % steps_per_update == 0:
                self.optim.step()
                self.optim.zero_grad()

            losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch}, Loss {loss.item()}")

            if self.args.scheduler == "cosine":
                self.lr_scheduler.step(
                    epoch + data_iter / len(dataloader), current_task=self.task_id
                )
            else:
                self.lr_scheduler.step(
                    data_iter,
                    len(dataloader) * sample_factor,
                    epoch,
                    current_task=self.task_id,
                )

            if self.args.scheduler != "cosine" and epoch * len(
                dataloader
            ) * sample_factor + data_iter == math.floor(
                self.args.total_epoch
                * len(dataloader)
                * sample_factor
                * self.args.constant_ratio
            ):
                self.save_annealed_model(task_id=self.task_id)
        avg_loss = sum(losses) / len(losses)
        logger.log(
            {
                "train/loss": avg_loss,
                "train/lr": self.optim.param_groups[0]["lr"],
                "train/epoch": epoch,
            }
        )

        return avg_loss

    def save_model(self,epoch,task_id):
        state_dict = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "task_id": task_id,
            "epoch": epoch,
        }
        # Check for previous checkpoints with the same task_id and delete them
        checkpoint_dir = self.args.output_dir
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith(f"mae-pretrain-{task_id}-") and filename.endswith(".pt"):
                logger.print(f"Warning: Removing previous checkpoint: {filename}")
                os.remove(os.path.join(checkpoint_dir, filename))

        torch.save(state_dict,f"{self.args.output_dir}/mae-pretrain-{task_id}-{epoch}.pt")
    def save_annealed_model(self,task_id):

        annealed_checkpoint_dir = f"{self.args.output_dir}/annealed"
        if not os.path.exists(annealed_checkpoint_dir):
            os.makedirs(annealed_checkpoint_dir)
        state_dict = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "task_id": task_id,
        }
        torch.save(state_dict,f"{annealed_checkpoint_dir}/mae-pretrain-{task_id}-annealed.pt")
        logger.print(f"Saved annealed model for task {task_id}")

    def finetune_epoch(self, dataloader, epoch, steps_per_update):
        self.finetune_model.train()
        losses = []
        step_count = 0
        self.finetune_optim.zero_grad()

        pbar = tqdm(dataloader,total=len(dataloader))
        for img, label in pbar:
            step_count += 1
            img = img.to(self.device)
            label = label.to(self.device)

            logits = self.finetune_model(img)
            loss = self.finetune_loss(logits, label)

            loss.backward()

            if step_count % steps_per_update == 0:
                self.finetune_optim.step()
                self.finetune_optim.zero_grad()

            losses.append(loss.item())
            pbar.set_description(f"Finetune Epoch {epoch}, Loss {loss.item()}")

        self.finetune_lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        logger.log({
            "finetune/loss": avg_loss,
            "finetune/lr": self.finetune_optim.param_groups[0]["lr"],
            "finetune/epoch": epoch,
        })

        return avg_loss
