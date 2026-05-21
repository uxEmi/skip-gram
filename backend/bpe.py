import re
from collections import Counter


_WORD_RE = re.compile(r"\w+|\S")


def pre_tokenize(text):
    return _WORD_RE.findall(text)


def _count_pairs(word_splits):
    pairs = Counter()
    for split, freq in word_splits.items():
        for i in range(len(split) - 1):
            pairs[(split[i], split[i + 1])] += freq
    return pairs


def _apply_merge(word_splits, pair):
    left, right = pair
    merged = left + right
    new_splits = {}
    for split, freq in word_splits.items():
        new_split = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and split[i] == left and split[i + 1] == right:
                new_split.append(merged)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        key = tuple(new_split)
        new_splits[key] = new_splits.get(key, 0) + freq
    return new_splits


def train(corpus, vocab_size=500):
    words = pre_tokenize(corpus)
    word_freqs = Counter(words)
    word_splits = {tuple(word): freq for word, freq in word_freqs.items()}

    vocab = set()
    for split in word_splits:
        vocab.update(split)

    merges = []
    while len(vocab) < vocab_size:
        pair_counts = _count_pairs(word_splits)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        word_splits = _apply_merge(word_splits, best_pair)
        merges.append(best_pair)
        vocab.add(best_pair[0] + best_pair[1])

    vocab_list = ["[UNK]"] + sorted(vocab)
    vocab_dict = {tok: i for i, tok in enumerate(vocab_list)}
    return {"vocab": vocab_dict, "merges": merges}


def encode(text, vocab, merges):
    token_ids = []
    for word in pre_tokenize(text):
        tokens = list(word)
        for left, right in merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(left + right)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        for tok in tokens:
            token_ids.append(vocab.get(tok, vocab["[UNK]"]))
    return token_ids


def decode(token_ids, vocab):
    id_to_token = {i: tok for tok, i in vocab.items()}
    return "".join(id_to_token.get(i, "") for i in token_ids)

