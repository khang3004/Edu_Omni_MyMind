from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import typer

from nlp_model_training.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, settings
from nlp_model_training.data.preprocess import clean_text

app = typer.Typer()


def generate_mock_nlp_data(output_path: Path) -> None:
    """
    Generates a mock classification dataset for demonstration and testing.
    """
    logger.info("Raw dataset not found. Generating mock sentiment analysis dataset...")

    data = {
        "text": [
            "I absolutely love this new neural network library! It works incredibly well.",
            "This model is terrible, it fails to converge and has horrible accuracy.",
            "An excellent paper on transformer architectures. Very clear explanation.",
            "What a complete waste of time. The results are totally non-reproducible.",
            "The dataset was structured beautifully, making tokenization very easy.",
            "Terrible documentation. I spent hours and still couldn't run the training loop.",
            "The attention mechanism is highly efficient and accelerates convergence.",
            "It crashed on the first epoch. Exception handling is nonexistent.",
            "Amazing accuracy on validation benchmarks. Truly state-of-the-art results.",
            "Do not buy this GPU. It overheats quickly and has poor support.",
        ]
        * 10,  # Create 100 rows
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10,
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.success(f"Mock raw dataset successfully generated at: {output_path}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """
    Cleans, normalizes, and splits raw text datasets for training.
    """
    logger.info("Executing NLP dataset preprocessing pipeline...")

    # Check if raw data exists, otherwise generate mock dataset
    if not input_path.exists():
        generate_mock_nlp_data(input_path)

    # Read raw data
    logger.info(f"Reading raw data from: {input_path}")
    df = pd.read_csv(input_path)

    # Verify columns
    if "text" not in df.columns or "label" not in df.columns:
        logger.critical("Input dataset must contain 'text' and 'label' columns!")
        raise ValueError("Invalid dataset schema.")

    # Apply text cleaning
    logger.info("Applying text normalization and cleaning...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Split into Train and Validation
    logger.info(f"Splitting dataset (test_size={settings.data.test_size}, seed={settings.seed})...")
    train_df, val_df = train_test_split(
        df,
        test_size=settings.data.test_size,
        random_state=settings.seed,
        stratify=df["label"],
    )

    # Save processed files
    output_dir = output_path.parent
    train_out = output_dir / "train.csv"
    val_out = output_dir / "val.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    logger.success("Data pipeline complete!")
    logger.info(f"Saved training partition ({len(train_df)} samples) to: {train_out}")
    logger.info(f"Saved validation partition ({len(val_df)} samples) to: {val_out}")


if __name__ == "__main__":
    app()
