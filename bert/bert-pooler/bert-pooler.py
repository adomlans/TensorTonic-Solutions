import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        cls_hidden = hidden_states[:, 0, :]
        pooled = cls_hidden @ self.W + self.b
        return tanh(pooled)

class SequenceClassifier:
    def __init__(self, hidden_size: int, num_classes: int):
        self.pooler = BertPooler(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        pooled = self.pooler.forward(hidden_states)
        return pooled @ self.classifier
