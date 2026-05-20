import re
import unicodedata

from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def clean_text(text: str) -> str:
    """
    Applies standard NLP cleaning and normalization to text.

    Args:
        text (str): The raw input string.

    Returns:
        str: The normalized, cleaned string.
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace multiple whitespaces and newlines with a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing spaces
    return text.strip()


def get_tokenizer(model_name: str = "distilbert-base-uncased") -> PreTrainedTokenizerFast:
    """
    Loads and caches a pretrained Hugging Face tokenizer.

    Args:
        model_name (str): The identifier of the tokenizer from huggingface.co.

    Returns:
        PreTrainedTokenizerFast: Fast tokenizer object.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer {model_name}. Error: {e}")
        raise e
