import json
import math

import numpy as np

import word2vec as w2v


def sanitize(curves):
    for k, curve in curves.items():
        last = curve[0]
        for i, v in enumerate(curve):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                curve[i] = last
            else:
                last = v
    return curves

LR_VALUES = [0.01, 0.05, 0.1, 0.5]
DIM_VALUES = [10, 50, 100, 200]


def train_once(token_ids, pairs, vocab_size, dim, lr, epochs, neg_samples, seed):
    sampler = w2v.NegativeSampler(token_ids, vocab_size, seed=seed)
    model = w2v.SkipGram(vocab_size, dim=dim, seed=seed)
    rng = np.random.default_rng(seed)
    pairs_local = pairs.copy()

    losses = []
    for epoch in range(epochs):
        rng.shuffle(pairs_local)
        total = 0.0
        for centre, context in pairs_local:
            negs = sampler.sample(neg_samples)
            targets = np.concatenate(([context], negs))
            v, u, s = model.forward(centre, targets)
            total += model.backward(centre, targets, v, u, s, lr=lr)
        losses.append(total / len(pairs_local))
    return losses


def main():
    corpus = w2v.load_corpus()
    token_ids, vocab, _ = w2v.tokenize_corpus(corpus)
    vocab_size = len(vocab)
    pairs = w2v.build_training_pairs(token_ids)

    print(f"corpus tokens: {len(token_ids)}  vocab: {vocab_size}  pairs: {len(pairs)}")
    print(f"epochs: {w2v.EPOCHS}  neg samples: {w2v.NEG_SAMPLES}  seed: {w2v.SEED}\n")

    lr_curves = {}
    for lr in LR_VALUES:
        print(f"=== LR = {lr} ===")
        lr_curves[str(lr)] = train_once(
            token_ids, pairs, vocab_size,
            dim=w2v.EMBED_DIM, lr=lr,
            epochs=w2v.EPOCHS, neg_samples=w2v.NEG_SAMPLES, seed=w2v.SEED,
        )
        print(f"  start {lr_curves[str(lr)][0]:.4f}  end {lr_curves[str(lr)][-1]:.4f}\n")

    dim_curves = {}
    for dim in DIM_VALUES:
        print(f"=== EMBED_DIM = {dim} ===")
        dim_curves[str(dim)] = train_once(
            token_ids, pairs, vocab_size,
            dim=dim, lr=w2v.LR,
            epochs=w2v.EPOCHS, neg_samples=w2v.NEG_SAMPLES, seed=w2v.SEED,
        )
        print(f"  start {dim_curves[str(dim)][0]:.4f}  end {dim_curves[str(dim)][-1]:.4f}\n")

    with open("comparisons.json", "w") as f:
        json.dump({
            "lr": {"values": LR_VALUES, "curves": sanitize(lr_curves)},
            "dim": {"values": DIM_VALUES, "curves": sanitize(dim_curves)},
            "epochs": w2v.EPOCHS,
        }, f)
    print("saved comparisons.json")


if __name__ == "__main__":
    main()

