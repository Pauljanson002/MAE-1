import os
from tokenize import group
import wandb
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
from typing import Dict, Any, Optional

import logging

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=SingletonType):
    def __init__(self):
        self._project_name = None
        self._experiment_name = None
        self._log_dir = None
        self._config = None
        self._tb_writer = None
        self._json_log = []
        self._json_log_path = None
        self._step = 0
        self.py_logger = logging.getLogger(__name__)

    def init(
        self,
        project_name: str,
        experiment_name: str,
        log_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._project_name = project_name
        self._experiment_name = experiment_name
        self._log_dir = Path(log_dir)
        self._config = vars(config) or {}
        self._step = 0  # Reset step counter

        # Path for wandb ID
        wandb_id_path = self._log_dir / "wandb_id.txt"
        wandb_id = None

        # Load existing wandb ID if resuming
        if wandb_id_path.exists():
            with open(wandb_id_path, "r") as f:
                wandb_id = f.read().strip()

        # Initialize wandb
        if self._experiment_name == "tmp" or "no_wandb" in self._experiment_name:
            wandb_mode = "disabled"
        else:
            wandb_mode = "online"

        # Split experiment name into tags
        tags = self._experiment_name.split("-") if self._experiment_name else []
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=self._config,
            mode=wandb_mode,
            dir=log_dir,
            resume="allow" if wandb_id else None,
            id=wandb_id,
            tags=tags,
            group=tags[0] if tags else None,
        )
        if wandb.run.sweep_id:
            def setup_for_sweep():
                self._experiment_name = f"{experiment_name}-{wandb.run.sweep_id}-{wandb.run.id}"
                self._log_dir = Path(log_dir) / self._experiment_name
                wandb.run.name = self._experiment_name   
            setup_for_sweep()
            wandb_id_path = self._log_dir / "wandb_id.txt"
        os.makedirs(self._log_dir, exist_ok=True)

        # Save the wandb ID for future resumption
        if wandb.run.id:
            with open(wandb_id_path, "w") as f:
                f.write(wandb.run.id)

        # Initialize TensorBoard
        self._tb_writer = SummaryWriter(log_dir=str(self._log_dir / "tensorboard"))

        # Initialize JSON logger
        self._json_log = []
        self._json_log_path = self._log_dir / f"{experiment_name}_log.json"

        # Load existing JSON log if resuming
        if self._json_log_path.exists():
            try:
                with open(self._json_log_path, "r") as f:
                    self._json_log = json.load(f)
                    # Update step to last logged step
                    if self._json_log:
                        self._step = self._json_log[-1]["step"] + 1
            except json.JSONDecodeError:
                print("Warning: Could not load existing JSON log")

        # Dump config as JSON
        config_json_path = self._log_dir / f"{self._experiment_name}_config.json"
        try:
            with open(config_json_path, "w") as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config as JSON: {e}")

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if self._project_name is None:
            raise RuntimeError("Logger not initialized. Call init() first.")

        # Use provided step or internal step
        current_step = step if step is not None else self._step

        # Log to wandb
        wandb.log(data, step=current_step)

        # Log to TensorBoard
        for key, value in data.items():
            self._tb_writer.add_scalar(key, value, global_step=current_step)

        # Log to JSON
        log_entry = {"step": current_step, **data}
        self._json_log.append(log_entry)

        # Increment internal step if no step was provided
        if step is None:
            self._step += 1

    def save_json_log(self):
        if self._json_log_path:
            with open(self._json_log_path, "w") as f:
                json.dump(self._json_log, f, indent=2)

    def finish(self):
        if self._project_name is None:
            return

        # Save wandb summary
        wandb_summary_path = self._log_dir / "wandb_summary.json"
        try:
            with open(wandb_summary_path, "w") as f:
                json.dump(dict(wandb.run.summary), f, indent=2)
        except Exception as e:
            print(f"Failed to save wandb summary: {e}")
        wandb.finish()
        if self._tb_writer:
            self._tb_writer.close()
        self.save_json_log()

        # Reset all variables
        self._project_name = None
        self._experiment_name = None
        self._log_dir = None
        self._config = None
        self._tb_writer = None
        self._json_log = []
        self._json_log_path = None
        self._step = 0

    def get_current_step(self):
        return self._step

    def set_step(self, step: int):
        self._step = step
    def print(self, msg: str):
        self.py_logger.info(msg)


# Example usage (if this file is run directly)
if __name__ == "__main__":
    logger = Logger()
    logger.init("MAE_Continual_Learning", "Experiment_1", "./logs")

    # Log some metrics
    for i in range(100):
        logger.log({"loss": 0.1 * i, "accuracy": i})  # No need to provide step

    # Finish logging
    logger.finish()
