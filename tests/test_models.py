import unittest
import torch
from nlp_model_training.models.registry import get_model
from nlp_model_training.models.architectures import LSTMClassifier


class TestModelRegistry(unittest.TestCase):
    def test_registry_contains_architectures(self):
        """
        Verify that both models are properly registered and loaded.
        """
        model_lstm = get_model(
            name="custom_lstm",
            vocab_size=1000,
            embedding_dim=64,
            hidden_size=64,
            num_labels=2,
        )
        self.assertIsInstance(model_lstm, LSTMClassifier)

    def test_lstm_forward_pass_dimensions(self):
        """
        Verify that a dummy forward pass through the LSTM model returns correct shapes.
        """
        batch_size = 4
        seq_len = 16
        num_labels = 3

        model = get_model(
            name="custom_lstm",
            vocab_size=100,
            embedding_dim=32,
            hidden_size=32,
            num_labels=num_labels,
        )

        # Build dummy input IDs tensor
        dummy_inputs = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)

        # Execute forward pass
        logits = model(input_ids=dummy_inputs)

        # Verify shape
        expected_shape = (batch_size, num_labels)
        self.assertEqual(logits.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
