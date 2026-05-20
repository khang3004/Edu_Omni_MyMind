from pathlib import Path

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlp_model_training.training.metrics import compute_classification_metrics


class Trainer:
    """
    Enterprise-grade custom PyTorch Trainer for NLP text classification tasks.
    Supports GPU/MPS acceleration, early stopping, validation tracking, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        learning_rate: float = 2.0e-5,
        weight_decay: float = 0.01,
        epochs: int = 5,
        device: torch.device | None = None,
        output_dir: str = "models/",
        use_early_stopping: bool = True,
        patience: int = 2,
    ):
        """
        Args:
            model (nn.Module): Registered PyTorch/Transformer model architecture.
            train_dataloader (DataLoader): PyTorch DataLoader for training data.
            val_dataloader (DataLoader, optional): DataLoader for validation evaluations.
            learning_rate (float): Optimizing step size.
            weight_decay (float): Regularizer weight decay value.
            epochs (int): Max number of times to iterate dataset.
            device (torch.device, optional): Chosen compute device.
            output_dir (str): Directory where best checkpoints are saved.
            use_early_stopping (bool): Activate early training termination if loss plateaus.
            patience (int): Iterations to wait for val loss improvements.
        """
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure optimizer (AdamW is standard for modern deep learning)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        # Early stopping configurations
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> float:
        """
        Executes a single training epoch across all training batches.
        """
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [Train]",
            leave=False,
        )

        for batch in progress_bar:
            # Transfer all batch tensors to target acceleration device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = self.criterion(logits, labels)

            # Backward pass & optimization step
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Runs evaluation on the model using a target DataLoader.
        """
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        # Concat outputs across all batches
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute metrics
        metrics = compute_classification_metrics(all_logits, all_labels)
        metrics["loss"] = avg_loss

        return metrics

    def fit(self) -> nn.Module:
        """
        Runs the full training process for the configured epochs.
        """
        logger.info("Initiated model training process...")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.4f}")

            if self.val_dataloader is not None:
                val_metrics = self.evaluate(self.val_dataloader)
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]
                val_f1 = val_metrics["macro_f1"]

                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Accuracy: {val_acc:.4f} | "
                    f"Val Macro-F1: {val_f1:.4f}"
                )

                # Dynamic early stopping & model checkpoint saving
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best_model.pt")
                else:
                    self.patience_counter += 1
                    logger.debug(
                        f"Val loss did not improve. Early stopping: {self.patience_counter}/{self.patience}"
                    )

                if self.use_early_stopping and self.patience_counter >= self.patience:
                    logger.warning(
                        f"Early stopping triggered! Training stopped after {epoch + 1} epochs."
                    )
                    break
            else:
                # Save checkpoints per epoch if no validation data is specified
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        logger.success("Training run completed successfully.")

        # Reload the best state dict before returning
        best_checkpoint_path = self.output_dir / "best_model.pt"
        if self.val_dataloader is not None and best_checkpoint_path.exists():
            self.load_checkpoint("best_model.pt")

        return self.model

    def save_checkpoint(self, filename: str) -> None:
        """
        Saves self state dict to output directory.
        """
        save_path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            save_path,
        )
        logger.info(f"Successfully serialized model checkpoint: {save_path}")

    def load_checkpoint(self, filename: str) -> None:
        """
        Loads self state dict from output directory.
        """
        load_path = self.output_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Loaded model state weights from: {load_path}")
