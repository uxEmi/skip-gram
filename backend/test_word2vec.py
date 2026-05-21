import numpy as np

try:
    import word2vec as w2v
    if not hasattr(w2v, "build_training_pairs"):
        raise AttributeError
except (ImportError, AttributeError, SyntaxError):
    print("word2vec.py is still the spec (no functions yet).")
    print("Start with PIECE 1: build_training_pairs. Then re-run me.")
    raise SystemExit


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    return cond


def test_pairs():
    ids = [10, 11, 12, 13, 14]
    try:
        p = w2v.build_training_pairs(ids, window=2)
    except NotImplementedError:
        return check("PIECE 1 build_training_pairs", False, "not implemented yet")
    s = {tuple(x) for x in p.tolist()}

    want_mid = {(12, 10), (12, 11), (12, 13), (12, 14)}

    want_edge = {(10, 11), (10, 12)}
    ok = want_mid <= s and want_edge <= s and (12, 12) not in s
    return check("PIECE 1 build_training_pairs", ok,
                 f"got {sorted(s)} — check edge clipping and skipping j==0")


def test_sampler():
    ids = np.array([0, 0, 0, 0, 1, 1, 2])
    try:
        smp = w2v.NegativeSampler(ids, vocab_size=3)
    except NotImplementedError:
        return check("PIECE 2 NegativeSampler.prob", False, "not implemented yet")
    p = smp.prob
    ok = (p is not None
          and abs(p.sum() - 1.0) < 1e-9
          and p[0] > p[1] > p[2]
          and abs(p[0] - (4**0.75) / (4**0.75 + 2**0.75 + 1**0.75)) < 1e-9)
    return check("PIECE 2 NegativeSampler.prob", ok,
                 "prob must be counts**0.75 normalised to sum 1")


def test_forward():
    m = w2v.SkipGram(vocab_size=5, dim=3, seed=1)
    m.W_in[2] = np.array([1.0, 0.0, 0.0])
    m.W_out[0] = np.array([1.0, 0.0, 0.0])
    m.W_out[1] = np.array([-1.0, 0.0, 0.0])
    try:
        v, u, s = m.forward(2, np.array([0, 1]))
    except NotImplementedError:
        return check("PIECE 3 SkipGram.forward", False, "not implemented yet")
    ok = (s is not None and s.shape == (2,)
          and abs(s[0] - 1 / (1 + np.exp(-1))) < 1e-6
          and abs(s[1] - 1 / (1 + np.exp(1))) < 1e-6)
    return check("PIECE 3 SkipGram.forward", ok,
                 "s[j] must be sigmoid(W_in[centre] . W_out[target_j])")


def test_backward():
    m = w2v.SkipGram(vocab_size=6, dim=4, seed=2)
    centre = 3
    targets = np.array([1, 4, 5])
    try:
        v, u, s = m.forward(centre, targets)


        v_c, u_c, s_c = v.copy(), u.copy(), s.copy()
        before_in = m.W_in[centre].copy()
        before_out = m.W_out[targets].copy()
        loss = m.backward(centre, targets, v, u, s, lr=0.1)
    except NotImplementedError:
        return check("PIECE 4 SkipGram.backward", False, "not implemented yet")


    moved = not np.allclose(before_out, m.W_out[targets])


    y = np.array([1.0, 0.0, 0.0])
    e = s_c - y
    exp_in = before_in - 0.1 * (e @ u_c)
    exp_out = before_out - 0.1 * np.outer(e, v_c)
    correct = (np.allclose(m.W_in[centre], exp_in, atol=1e-8)
               and np.allclose(m.W_out[targets], exp_out, atol=1e-8))
    return check("PIECE 4 SkipGram.backward", moved and correct,
                 "update must be  W -= lr * grad  with e = s - y")


def test_end_to_end():
    try:
        model, vocab, id2t, hist = w2v.train_model()
    except NotImplementedError:
        return check("END-TO-END training", False, "finish the 4 pieces first")
    dropped = hist[0] - hist[-1]
    ok = dropped > 0.05 and hist[-1] < hist[0]
    print(f"       loss {hist[0]:.3f} -> {hist[-1]:.3f}")
    if ok:
        print("       neighbours of 'king':",
              w2v.nearest(model, vocab, id2t, "king", top=5))
    return check("END-TO-END loss decreases", ok)


if __name__ == "__main__":
    print("Implement the pieces in order. Each unlocks the next.\n")
    if not test_pairs():
        raise SystemExit
    if not test_sampler():
        raise SystemExit
    if not test_forward():
        raise SystemExit
    if not test_backward():
        raise SystemExit
    print()
    test_end_to_end()
    print("\nAll green? You understand Word2Vec. Now you can explain every line.")

