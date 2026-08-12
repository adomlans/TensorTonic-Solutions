import numpy as np
from typing import List

class MockBertEncoder:
    def __init__(self, hidden_size: int, num_layers: int):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_frozen = [False] * num_layers

    def freeze_layers(self, layer_indices: List[int]):
        for idx in layer_indices:
            self.layer_frozen[idx] = True

    def unfreeze_all(self):
        self.layer_frozen = [False] * self.num_layers

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        x = embeddings
        for i, layer in enumerate(self.layers):
            x = x @ layer + x
        return x

class BertForSequenceClassification:
    def __init__(self, hidden_size: int, num_labels: int, num_layers: int = 3):
        self.encoder = MockBertEncoder(hidden_size, num_layers)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        hidden = self.encoder.forward(embeddings)
        cls_hidden = hidden[:, 0, :]
        return cls_hidden @ self.classifier

class BertForTokenClassification:
    def __init__(self, hidden_size: int, num_labels: int, num_layers: int = 3):
        self.encoder = MockBertEncoder(hidden_size, num_layers)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        hidden = self.encoder.forward(embeddings)
        return hidden @ self.classifier

