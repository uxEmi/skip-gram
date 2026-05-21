import json

import numpy as np

import bpe


EMBED_DIM = 50
WINDOW = 2
NEG_SAMPLES = 5
EPOCHS = 50
LR = 0.05
BPE_VOCAB_SIZE = 600
SEED = 0


def load_corpus(path="corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize_corpus(corpus):
    model = bpe.train(corpus, vocab_size=BPE_VOCAB_SIZE)
    vocab = model["vocab"]
    ids = bpe.encode(corpus, vocab, model["merges"])
    id_to_token = {i: t for t, i in vocab.items()}
    return ids, vocab, id_to_token


def build_training_pairs(token_ids, window=WINDOW):
    pairs = []
    n = len(token_ids)
    for i in range(n):
        for j in range(-window, window + 1):
            if j == 0:
                continue
            k = i + j
            if k < 0 or k >= n:
                continue
            pairs.append([token_ids[i], token_ids[k]])
    return np.array(pairs, dtype=np.int64)


class NegativeSampler:

    def __init__(self, token_ids, vocab_size, power=0.75, seed=SEED):
        self.rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size

        counts = np.bincount(token_ids, minlength=vocab_size).astype(np.float64)
        weighted = counts ** power
        self.prob = weighted / weighted.sum()

    def sample(self, n):
        return self.rng.choice(self.vocab_size, size=n, p=self.prob)


def sigmoid(x):

    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SkipGram:

    def __init__(self, vocab_size, dim=EMBED_DIM, seed=SEED):
        rng = np.random.default_rng(seed)

        self.W_in = (rng.random((vocab_size, dim)) - 0.5) / dim
        self.W_out = np.zeros((vocab_size, dim))

    def forward(self, centre, targets):
        v = self.W_in[centre].copy()
        u = self.W_out[targets].copy()
        s = sigmoid(u @ v)
        return v, u, s

    def backward(self, centre, targets, v, u, s, lr=LR):
        y = np.zeros(len(targets))
        y[0] = 1.0
        e = s - y
        grad_in = e @ u
        grad_out = np.outer(e, v)
        self.W_out[targets] -= lr * grad_out
        self.W_in[centre] -= lr * grad_in
        eps = 1e-10
        return -np.log(s[0] + eps) - np.sum(np.log(1.0 - s[1:] + eps))


def train_model(verbose=True):
    corpus = load_corpus()
    token_ids, vocab, id_to_token = tokenize_corpus(corpus)
    vocab_size = len(vocab)
    pairs = build_training_pairs(token_ids)
    sampler = NegativeSampler(token_ids, vocab_size)
    model = SkipGram(vocab_size)

    if verbose:
        print(f"corpus tokens : {len(token_ids)}")
        print(f"vocab size    : {vocab_size}")
        print(f"training pairs: {len(pairs)}\n")

    rng = np.random.default_rng(SEED)
    loss_history = []
    for epoch in range(EPOCHS):
        rng.shuffle(pairs)
        total = 0.0
        for centre, context in pairs:
            negs = sampler.sample(NEG_SAMPLES)
            targets = np.concatenate(([context], negs))
            v, u, s = model.forward(centre, targets)
            total += model.backward(centre, targets, v, u, s)
        avg = total / len(pairs)
        loss_history.append(avg)
        if verbose:
            print(f"epoch {epoch + 1:3d}/{EPOCHS}   avg loss {avg:.4f}")
    return model, vocab, id_to_token, loss_history


def save_artifacts(model, vocab, id_to_token, loss_history,
                    model_path="model.npz", meta_path="vocab.json"):
    np.savez(model_path, W_in=model.W_in)
    with open(meta_path, "w") as f:
        json.dump({
            "vocab": vocab,
            "id_to_token": {str(i): t for i, t in id_to_token.items()},
            "loss_history": loss_history,
        }, f)


def nearest(model, vocab, id_to_token, token, top=8):
    if token not in vocab:
        return f"'{token}' is not in the vocabulary"
    W = model.W_in
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-10)
    q = Wn[vocab[token]]
    sims = Wn @ q
    order = np.argsort(-sims)
    return [(id_to_token[i], round(float(sims[i]), 3))
            for i in order if i != vocab[token]][:top]


if __name__ == "__main__":
    model, vocab, id_to_token, loss_history = train_model()
    save_artifacts(model, vocab, id_to_token, loss_history)
    print("\nsaved model.npz + vocab.json")
    print("\nnearest neighbours:")
    for w in ["king", "queen", "river", "france"]:
        print(f"  {w:>7} -> {nearest(model, vocab, id_to_token, w, top=6)}")

