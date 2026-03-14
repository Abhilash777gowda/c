import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils.helpers import setup_logging, get_device
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

logger = setup_logging()

# FastText English vectors (Common Crawl, 300-dim)
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
FASTTEXT_PATH = "models/fasttext/cc.en.300.bin"
EMBED_DIM = 300  # FastText dimension


class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        encoded_texts = []
        for text in texts:
            tokens = text.split()
            encoded = [vocab.get(token, vocab['__UNK__']) for token in tokens]
            if len(encoded) < max_len:
                encoded = encoded + [vocab['__PAD__']] * (max_len - len(encoded))
            else:
                encoded = encoded[:max_len]
            encoded_texts.append(encoded)
        self.texts = torch.tensor(encoded_texts, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Load FastText weights if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # freeze during early training
            logger.info("FastText embeddings loaded into BiLSTM embedding layer.")

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        # Concat final forward + backward hidden state from top layer
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.dropout(hidden))


def _try_load_fasttext(vocab: dict) -> torch.Tensor | None:
    """
    Load FastText pre-trained embeddings for vocabulary words.
    Downloads the binary if not cached. Returns embedding matrix or None.
    """
    try:
        import fasttext  # pip install fasttext-wheel
    except ImportError:
        logger.warning("fasttext package not installed. Run: pip install fasttext-wheel")
        return None

    if not os.path.exists(FASTTEXT_PATH):
        logger.info("FastText binary not found locally. Attempting download (this may take a while ~4GB)...")
        try:
            import urllib.request, gzip, shutil
            os.makedirs(os.path.dirname(FASTTEXT_PATH), exist_ok=True)
            gz_path = FASTTEXT_PATH + ".gz"
            urllib.request.urlretrieve(FASTTEXT_URL, gz_path)
            with gzip.open(gz_path, 'rb') as f_in, open(FASTTEXT_PATH, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
            logger.info("FastText binary downloaded and extracted.")
        except Exception as e:
            logger.warning(f"FastText download failed: {e}. Using random embeddings.")
            return None

    try:
        ft = fasttext.load_model(FASTTEXT_PATH)
        vocab_size = len(vocab)
        embedding_matrix = np.zeros((vocab_size, EMBED_DIM), dtype=np.float32)
        for word, idx in vocab.items():
            if word not in ('__PAD__', '__UNK__'):
                embedding_matrix[idx] = ft.get_word_vector(word)
        logger.info(f"FastText embedding matrix built for {vocab_size} vocab tokens.")
        return torch.from_numpy(embedding_matrix)
    except Exception as e:
        logger.warning(f"FastText loading failed: {e}. Using random embeddings.")
        return None


class CustomBiLSTMClassifier:
    def __init__(self, categories, vocab_size=5000, embed_dim=EMBED_DIM,
                 hidden_dim=256, use_fasttext=True):
        self.categories = categories
        self.device = get_device()
        self.vocab = {'__PAD__': 0, '__UNK__': 1}
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.use_fasttext = use_fasttext
        # Model is built after vocabulary construction
        self.model = None

    def build_vocab(self, texts, max_vocab_size=5000):
        from collections import Counter
        all_words = ' '.join(texts).split()
        counter = Counter(all_words)
        for word, _ in counter.most_common(max_vocab_size - 2):
            self.vocab[word] = len(self.vocab)

    def _build_model(self):
        pretrained = _try_load_fasttext(self.vocab) if self.use_fasttext else None
        actual_embed_dim = EMBED_DIM if pretrained is not None else 100
        self.model = BiLSTMModel(
            vocab_size=len(self.vocab),
            embed_dim=actual_embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.categories),
            pretrained_embeddings=pretrained,
        ).to(self.device)

    def train(self, df, epochs=3, batch_size=32, lr=1e-3):
        logger.info("Training BiLSTM Model (with FastText embeddings if available)...")
        if len(self.vocab) <= 2:
            self.build_vocab(df['clean_text'].tolist())

        if self.model is None:
            self._build_model()

        dataset = BiLSTMDataset(df['clean_text'].tolist(),
                                df[self.categories].values, self.vocab)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(
                f"BiLSTM Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}"
            )

    def evaluate(self, df):
        logger.info("Evaluating BiLSTM Model...")
        if self.model is None:
            logger.error("Model not trained yet.")
            return {}

        dataset = BiLSTMDataset(df['clean_text'].tolist(),
                                df[self.categories].values, self.vocab)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                outputs = self.model(batch_texts)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        h_loss = hamming_loss(all_labels, all_preds)

        print("\n--- BiLSTM Evaluation ---")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
              f"F1: {f1:.4f} | Hamming Loss: {h_loss:.4f}")
        return {"precision": precision, "recall": recall, "f1": f1, "hamming_loss": h_loss}
