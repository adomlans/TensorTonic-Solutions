import numpy as np

class BertEmbeddings:
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02

    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = token_ids.shape
        tok_emb = self.token_embeddings[token_ids]
        positions = np.arange(seq_len)
        pos_emb = self.position_embeddings[positions]
        seg_emb = self.segment_embeddings[segment_ids]
        return tok_emb + pos_emb + seg_emb
