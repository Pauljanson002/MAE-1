import math
from matplotlib.pylab import f
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from infinite_scheduler import InfiniteScheduler,CosineScheduler

from metrics import calculate_per_task_accuracy
from logger import Logger
logger = Logger()


class MAETrainer:
    def __init__(self, model, args):
        self.model = torch.compile(model)
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.amp.GradScaler()
        self.writer = SummaryWriter(os.path.join("logs", "cifar10", "mae-pretrain"))
        self.task_id = 0
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        if self.args.scheduler == "cosine":
            self.lr_scheduler = CosineScheduler(self.optim, self.args)
        else:
            self.lr_scheduler = InfiniteScheduler(self.optim, self.args)

    def _lr_func(self, epoch):
        return min(
            (epoch + 1) / (self.args.warmup_epoch + 1e-8),
            0.5 * (math.cos(epoch / self.args.total_epoch * math.pi) + 1),
        )
    def _lr_func_finetune(self, epoch):
        return min(
            (epoch + 1) / (self.args.finetune_warmup_epoch + 1e-8),
            0.5 * (math.cos(epoch / self.args.finetune_epoch * math.pi) + 1),
        )

    def train_epoch(self, dataloader, epoch, steps_per_update):
        self.model.train()
        losses = []
        step_count = 0
        self.optim.zero_grad()
        sample_factor = 1 if self.task_id == 0 else 2
        pbar = tqdm(dataloader, total=len(dataloader) * sample_factor)

        for data_iter, (img, _) in enumerate(pbar):
            step_count += 1

            with torch.amp.autocast("cuda"):
                predicted_img, mask = self.model(img)
                loss = (
                    torch.mean((predicted_img - img) ** 2 * mask) / self.args.mask_ratio
                )

            self.scaler.scale(loss).backward()

            if step_count % steps_per_update == 0:
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()

            losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch}, Loss {loss.item()}")

            if self.args.scheduler == "cosine":
                self.lr_scheduler.step(epoch +  data_iter / len(dataloader), current_task=self.task_id)
            else:
                self.lr_scheduler.step(
                    data_iter,
                    len(dataloader) * sample_factor,
                    epoch,
                    current_task=self.task_id,
                )

            if self.args.scheduler != "cosine" and epoch * len(dataloader) * sample_factor + data_iter == math.floor(self.args.total_epoch * len(dataloader) * sample_factor * self.args.constant_ratio):
                self.save_annealed_model(task_id=self.task_id)
        avg_loss = sum(losses) / len(losses)
        logger.log({
            "train/loss": avg_loss,
            "train/lr": self.optim.param_groups[0]["lr"],
            "train/epoch": epoch,
        })

        return avg_loss

    def save_model(self,epoch,task_id):
        state_dict = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "task_id": task_id,
            "scaler": self.scaler.state_dict(),
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
            "scaler": self.scaler.state_dict(),
        }
        torch.save(state_dict,f"{annealed_checkpoint_dir}/mae-pretrain-{task_id}-annealed.pt")
        logger.print(f"Saved annealed model for task {task_id}")

    def load_annealed_model(self,task_id):
        annealed_checkpoint_dir = f"{self.args.output_dir}/annealed"
        state_dict = torch.load(f"{annealed_checkpoint_dir}/mae-pretrain-{task_id}-annealed.pt")
        self.model.load_state_dict(state_dict["model"])
        logger.print(f"Loaded annealed model for task {task_id}")

    def freeze_model(self):
        logger.print("Freezing model")
        for param in self.model.parameters():
            param.requires_grad = False
    def unfreeze_model(self):
        logger.print("Unfreezing model")
        for param in self.model.parameters():
            param.requires_grad = True

    def set_finetune_model(self,finetune_model):
        logger.print("Setting up finetune model")
        self.finetune_model = torch.compile(finetune_model)
        self.finetune_optim = torch.optim.AdamW(
            self.finetune_model.parameters(),
            lr=self.args.finetune_learning_rate * self.args.finetune_batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        self.finetune_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.finetune_optim, lr_lambda=self._lr_func_finetune,
        )
        self.finetune_scaler = torch.amp.GradScaler()
        self.finetune_loss = torch.nn.CrossEntropyLoss()

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

            with torch.amp.autocast("cuda"):
                logits = self.finetune_model(img)
                loss = self.finetune_loss(logits, label)

            self.finetune_scaler.scale(loss).backward()

            if step_count % steps_per_update == 0:
                self.finetune_scaler.step(self.finetune_optim)
                self.finetune_scaler.update()
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

    def start_evaluation(self,eval_dataloader,task_id):
        self.model.eval()
        self.finetune_model.eval()

        with torch.no_grad(),torch.autocast("cuda"):
            predictions = []
            labels = []
            for img, label in eval_dataloader:
                img = img.to(self.device)
                label = label.to(self.device)

                logits = self.finetune_model(img)
                loss = self.finetune_loss(logits,label)
                predictions.append(logits.argmax(dim=-1))
                labels.append(label)
            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
            acc = torch.mean((predictions == labels).float())
            logger.print(f"Validation accuracy: {acc.item()}")
            metric_dict = calculate_per_task_accuracy(predictions,labels,current_task=task_id)

            metric_dict["valid/loss"] = loss.item()
            metric_dict["valid/acccuracy"] = acc.item()
            metric_dict["task_id"] = task_id
            logger.log(metric_dict)
