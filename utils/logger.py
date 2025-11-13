# utils/logger.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """
    Простая обёртка над SummaryWriter, которую использует train_cartpole.
    """

    def __init__(self, log_dir: str, flush_secs: int = 10, comment: str = ""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs,
            comment=comment,
        )

    # базовый метод
    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    # алиас, если вдруг в коде используется другое имя
    def scalar(self, tag: str, value: float, step: int):
        self.log_scalar(tag, value, step)

    # на всякий случай: пакет логов разом
    def log_scalars(self, main_tag: str, values: Dict[str, Any], step: int):
        """
        Log a dict of scalars under one main tag:
        e.g. main_tag="train", values={"loss": ..., "epsilon": ...}
        """
        self.writer.add_scalars(main_tag, values, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


__all__ = ["TBLogger"]
