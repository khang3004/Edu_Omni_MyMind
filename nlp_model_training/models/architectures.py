import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from nlp_model_training.models.registry import register_model


@register_model("transformer")
class TransformerClassifier(nn.Module):
    """
    Enterprise-ready custom sequence classifier wrapping a Hugging Face Transformer backbone
    with a robust classification head.
    """

    def __init__(
        self,
        pretrained_model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)

        hidden_size = self.config.hidden_size
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Forward pass through Hugging Face Transformer backbone
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if "token_type_ids" in self.config.model_type else None,
        )

        # Grab last hidden states of CLS token (index 0) representing the sequence embedding
        last_hidden_state = outputs[0]  # Shape: (batch_size, seq_len, hidden_size)
        cls_representation = last_hidden_state[:, 0]  # Shape: (batch_size, hidden_size)

        # Pass CLS representation through MLP classification head
        x = self.pre_classifier(cls_representation)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


@register_model("custom_lstm")
class LSTMClassifier(nn.Module):
    """
    A lightweight, robust bidirectional LSTM baseline for text classification.
    Ideal for fast iteration, CPU prototyping, or low-latency production applications.
    """

    def __init__(
        self,
        vocab_size: int = 30522,  # Default size of standard BERT uncased vocabulary
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_labels: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # input_ids: Shape (batch_size, sequence_length)
        embeddings = self.embedding(input_ids)  # Shape (batch_size, seq_len, embed_dim)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)  # Shape (batch_size, seq_len, hidden_size * 2)

        # Global mean pooling over sequence length
        pooled = torch.mean(lstm_out, dim=1)  # Shape (batch_size, hidden_size * 2)
        pooled = self.dropout(pooled)

        logits = self.fc(pooled)
        return logits
