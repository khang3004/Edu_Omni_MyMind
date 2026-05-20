from loguru import logger
import pandas as pd
import typer

from nlp_model_training.config import PROCESSED_DATA_DIR, settings
from nlp_model_training.data.dataset import build_dataloader
from nlp_model_training.data.preprocess import get_tokenizer
from nlp_model_training.models.registry import get_model
from nlp_model_training.training.trainer import Trainer
from nlp_model_training.utils.helpers import get_device, set_seed

app = typer.Typer()


@app.command()
def main():
    """
    Launches the training and validation run for the selected NLP model.
    """
    logger.info("Initializing NLP Model Training run...")

    # 1. Set seed for mathematical reproducibility
    set_seed(settings.seed)

    # 2. Select execution device (GPU, MPS Apple Silicon, or CPU)
    device = get_device()

    # 3. Load processed train/val dataset partitions
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"

    if not train_path.exists() or not val_path.exists():
        logger.error(
            f"Processed data not found at {PROCESSED_DATA_DIR}. "
            "Please run 'make data' or 'python nlp_model_training/dataset.py' first!"
        )
        raise FileNotFoundError("Missing training partitions. Run preprocessing first.")

    logger.info(f"Loading datasets from {PROCESSED_DATA_DIR}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Fill empty text columns if any
    train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")
    val_df["cleaned_text"] = val_df["cleaned_text"].fillna("")

    # 4. Load fast tokenizer from Hugging Face
    tokenizer = get_tokenizer(settings.model.pretrained_model_name)

    # 5. Build optimized PyTorch DataLoaders
    logger.info("Building PyTorch DataLoaders...")
    train_loader = build_dataloader(
        texts=train_df["cleaned_text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=settings.data.max_length,
        batch_size=settings.data.batch_size,
        shuffle=True,
        num_workers=settings.data.num_workers,
    )

    val_loader = build_dataloader(
        texts=val_df["cleaned_text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=settings.data.max_length,
        batch_size=settings.data.batch_size,
        shuffle=False,
        num_workers=settings.data.num_workers,
    )

    # 6. Instantiate model dynamically from registry
    logger.info(f"Loading registered model type: '{settings.model.model_type}'")
    model = get_model(
        name=settings.model.model_type,
        pretrained_model_name=settings.model.pretrained_model_name,
        num_labels=settings.model.num_labels,
        hidden_size=settings.model.hidden_size,
        dropout=settings.model.dropout,
    )

    # 7. Initialize and start Trainer loop
    logger.info("Initializing custom Trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
        epochs=settings.training.epochs,
        device=device,
        output_dir=settings.training.output_dir,
        use_early_stopping=settings.training.use_early_stopping,
        patience=settings.training.patience,
    )

    # Run fit
    trainer.fit()

    # 8. Run final evaluation on the best checkpoint
    logger.info("Running final evaluation of the best model checkpoint on validation set...")
    val_metrics = trainer.evaluate(val_loader)

    logger.success("=======================================================================")
    logger.success("TRAINING RUN RESULTS ON VALIDATION SET:")
    logger.success(f"Best Val Loss:       {val_metrics['loss']:.4f}")
    logger.success(f"Best Val Accuracy:   {val_metrics['accuracy']:.4f}")
    logger.success(f"Best Val Macro-F1:   {val_metrics['macro_f1']:.4f}")
    logger.success(f"Best Val Weighted-F1:{val_metrics['weighted_f1']:.4f}")
    logger.success("=======================================================================")


if __name__ == "__main__":
    app()
