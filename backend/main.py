import json
import os

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_W = np.load("model.npz")["W_in"]
with open("vocab.json") as f:
    _meta = json.load(f)
_vocab = _meta["vocab"]
_id2t = {int(i): t for i, t in _meta["id_to_token"].items()}
_loss = _meta["loss_history"]
_Wn = _W / (np.linalg.norm(_W, axis=1, keepdims=True) + 1e-10)


_centered = _W - _W.mean(axis=0, keepdims=True)
_, _, _Vt = np.linalg.svd(_centered, full_matrices=False)
_W_3d = _centered @ _Vt[:3].T
_W_3d = _W_3d / (np.abs(_W_3d).max() + 1e-10)



@app.get("/health")
def health():
    return {"status": "ok", "vocab_size": len(_vocab), "dim": int(_W.shape[1])}


@app.get("/neighbours")
def neighbours(word: str, top: int = 8):
    if word not in _vocab:
        return {"word": word, "found": False, "neighbours": []}
    q = _Wn[_vocab[word]]
    sims = _Wn @ q
    order = np.argsort(-sims)
    out = [{"token": _id2t[int(i)], "score": round(float(sims[i]), 4)}
           for i in order if int(i) != _vocab[word]][:top]
    return {"word": word, "found": True, "neighbours": out}


@app.get("/analogy")
def analogy(a: str, b: str, c: str, top: int = 8):
    missing = [w for w in (a, b, c) if w not in _vocab]
    if missing:
        return {"a": a, "b": b, "c": c, "found": False, "missing": missing, "analogy": []}
    q = _Wn[_vocab[a]] - _Wn[_vocab[b]] + _Wn[_vocab[c]]
    q = q / (np.linalg.norm(q) + 1e-10)
    sims = _Wn @ q
    excluded = {_vocab[a], _vocab[b], _vocab[c]}
    order = np.argsort(-sims)
    out = [{"token": _id2t[int(i)], "score": round(float(sims[i]), 4)}
           for i in order if int(i) not in excluded][:top]
    return {"a": a, "b": b, "c": c, "found": True, "analogy": out}


@app.get("/loss")
def loss():
    return {"loss_history": _loss,
            "start": _loss[0] if _loss else None,
            "end": _loss[-1] if _loss else None}


@app.get("/comparisons")
def comparisons():
    if not os.path.exists("comparisons.json"):
        return {"error": "run python generate_comparisons.py first"}
    with open("comparisons.json") as f:
        return json.load(f)


@app.get("/embedding_3d")
def embedding_3d():
    points = [
        {"token": _id2t[i],
         "x": float(_W_3d[i, 0]),
         "y": float(_W_3d[i, 1]),
         "z": float(_W_3d[i, 2])}
        for i in range(_W_3d.shape[0])
    ]
    return {"points": points, "dim": int(_W.shape[1])}

