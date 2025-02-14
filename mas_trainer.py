import torch
from tqdm import tqdm
from trainer import MAETrainer
from logger import Logger

logger = Logger()

class MASTrainer(MAETrainer):

    def __init__(self, model, args):
        super().__init__(model, args)
        self.alpha = 0.5
        self._lambda = 1.0

    def _get_importance(self,dataloader):
        # Get named parameters
        importance = {}
        for name, param in self.model.encoder.named_parameters():
            importance[name] = torch.zeros_like(param)
        for img, _ in dataloader:
            self.optim.zero_grad()
            with torch.amp.autocast("cuda"):
                features, backward_indexes = self.model.encoder(img)
                loss = torch.norm(features,p="fro",dim=1).pow(2).mean()
            self.scaler.scale(loss).backward()
            for name, param in self.model.encoder.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        importance[name] += torch.abs(param.grad)
                        
        # Normalize importance
        for name in importance.keys():
            importance[name] = importance[name] / float(len(dataloader))
        return importance
    
    
    def train_epoch(self, dataloader, epoch, steps_per_update):
        self.model.train()
        losses = []
        step_count = 0
        self.optim.zero_grad()

        pbar = tqdm(dataloader,total=len(dataloader))
        for img, _ in pbar:
            step_count += 1

            with torch.amp.autocast("cuda"):
                predicted_img, mask = self.model(img)
                loss = (
                    torch.mean((predicted_img - img) ** 2 * mask) / self.args.mask_ratio
                )

            self.before_backward(self.task_id,loss)
            self.scaler.scale(loss).backward()

            if step_count % steps_per_update == 0:
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()

            losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch}, Loss {loss.item()}")

        if self.args.scheduler == "cosine":
            self.lr_scheduler.step()
        else:
            self.lr_scheduler.step(epoch, current_task=self.task_id)
        avg_loss = sum(losses) / len(losses)
        logger.log({
            "train/loss": avg_loss,
            "train/lr": self.optim.param_groups[0]["lr"],
            "train/epoch": epoch,
        })

        return avg_loss
    
    
    def after_training_exp(self,task_id,train_dataloader):
        
        self.params_copy = {name: param.clone().detach() for name, param in self.model.encoder.named_parameters()}
        if task_id == 0:
            self.importance = self._get_importance(train_dataloader)
            return
        else:
            curr_importance = self._get_importance(train_dataloader)
        
        if not self.importance:
            raise ValueError("Importance not initialized")

        for name in curr_importance.keys():
            if name not in self.importance:
                self.importance[name] = curr_importance[name]
            else:
                self.importance[name] = self.alpha * self.importance[name] + (1 - self.alpha) * curr_importance[name]
    
    def before_backward(self,task_id,loss):
        if task_id == 0:
            return loss
        loss_reg = 0
        for name, param in self.model.encoder.named_parameters():
            if name in self.importance.keys():
                loss_reg += torch.sum(self.importance[name] * (param - self.params_copy[name]) ** 2)
        loss.add_(self._lambda * loss_reg)
    
        

        