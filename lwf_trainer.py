import math
import torch
from tqdm import tqdm
from trainer import MAETrainer
from logger import Logger
import copy
logger = Logger()


class LWFTrainer(MAETrainer):

    def __init__(self, model, args):
        super().__init__(model, args)
        self.alpha = 1.0
        self.prev_model_encoder = None

    def after_training_exp(self,task_id, train_dataloader):
        self.prev_model_encoder = copy.deepcopy(self.model.encoder)
        for param in self.prev_model_encoder.parameters():
            param.requires_grad = False

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
            self.before_backward(img,loss)
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

    def before_backward(self,img_batch,loss):
        if self.prev_model_encoder is None:
            return

        with torch.no_grad():
            prev_features,_ = self.prev_model_encoder(img_batch)

        new_features,_ = self.model.encoder(img_batch)

        distillation_loss = 1 - torch.nn.functional.cosine_similarity(prev_features,new_features).mean()
        loss.add_(self.alpha * distillation_loss)
