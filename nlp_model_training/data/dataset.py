import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from nlp_model_training.data.preprocess import clean_text


class TextClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for NLP text classification task.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        tokenizer: PreTrainedTokenizerFast | None = None,
        max_length: int = 128,
        do_clean: bool = True,
    ):
        """
        Args:
            texts (List[str]): List of input text documents.
            labels (List[int], optional): Categorical labels (0, 1, etc.). None for inference.
            tokenizer (PreTrainedTokenizerFast): Hugging Face tokenizer instance.
            max_length (int): Max sequence length for padding/truncating.
            do_clean (bool): Whether to clean text before tokenization.
        """
        self.texts = [clean_text(t) if do_clean else t for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize on-the-fly or return dictionary of input IDs
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # Remove batch dimension added by return_tensors='pt'
            item = {key: val.squeeze(0) for key, val in encoding.items()}
        else:
            item = {"text": text}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def build_dataloader(
    texts: list[str],
    labels: list[int] | None = None,
    tokenizer: PreTrainedTokenizerFast | None = None,
    max_length: int = 128,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Builds a PyTorch DataLoader from lists of texts and labels.

    Returns:
        DataLoader: PyTorch DataLoader for iteration.
    """
    dataset = TextClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
