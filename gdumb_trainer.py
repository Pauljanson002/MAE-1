import torch
from tqdm import tqdm
from model import MAE_ViT
from trainer import MAETrainer
from logger import Logger
import copy

logger = Logger()


class GDUMBTrainer(MAETrainer):

    def __init__(self, model, args):
        super().__init__(model, args)

    def after_finetuning(self):
        self.model = MAE_ViT(mask_ratio=self.args.mask_ratio).to("cuda")

