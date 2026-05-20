import unittest
import torch
from transformers import PreTrainedTokenizerFast
from nlp_model_training.data.preprocess import clean_text, get_tokenizer
from nlp_model_training.data.dataset import TextClassificationDataset

class TestDataPipeline(unittest.TestCase):
    def test_clean_text_lowercasing_and_strip(self):
        """
        Tests that text normalization correctly lowercases and strips whitespace.
        """
        raw_text = "   Transformer Model training is AWESOME! \n  "
        expected = "transformer model training is awesome!"
        self.assertEqual(clean_text(raw_text), expected)

    def test_clean_text_unicode_normalization(self):
        """
        Tests that text normalization properly standardizes unicode variants.
        """
        # Testing NFKC standardizations (e.g. double spaces)
        raw_text = "This is a “test” with double  spaces."
        expected = "this is a “test” with double spaces."
        self.assertEqual(clean_text(raw_text), expected)

    def test_dataset_item_without_tokenizer(self):
        """
        Tests PyTorch Dataset mapping when no tokenizer is supplied.
        """
        texts = ["Sentence A", "Sentence B"]
        labels = [1, 0]
        dataset = TextClassificationDataset(texts=texts, labels=labels, tokenizer=None, do_clean=True)
        
        self.assertEqual(len(dataset), 2)
        
        item = dataset[0]
        self.assertEqual(item["text"], "sentence a")
        self.assertEqual(item["labels"].item(), 1)
        self.assertIsInstance(item["labels"], torch.Tensor)

if __name__ == "__main__":
    unittest.main()
