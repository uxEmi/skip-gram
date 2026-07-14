# w2v — Word2Vec from scratch

A small, self-contained word2vec implementation with an interactive frontend. Embeddings are trained in NumPy (skip-gram with negative sampling) over a BPE-tokenized corpus, then served through a FastAPI backend and explored in a React app that renders the vectors in 3D.

## Authors

- Mihai
- Ayla

## What's inside

- **BPE tokenizer** (`backend/bpe.py`) — byte-pair encoding trained on the corpus (`BPE_VOCAB_SIZE = 600`).
- **Word2Vec trainer** (`backend/word2vec.py`) — skip-gram with negative sampling, written in plain NumPy.
  - `EMBED_DIM = 50`, `WINDOW = 2`, `NEG_SAMPLES = 5`, `EPOCHS = 50`, `LR = 0.05`
  - Negative sampling uses the unigram distribution raised to the `0.75` power.
- **API** (`backend/main.py`) — FastAPI service that loads the trained model and exposes it.
- **Frontend** (`frontend/`) — React + Vite app using `@react-three/fiber` to plot the embedding space in 3D.

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Vocab size and embedding dimension. |
| `GET /neighbours?word=<w>&top=<n>` | Nearest neighbours by cosine similarity. |
| `GET /loss` | Training loss history (start / end). |
| `GET /comparisons` | Precomputed word comparisons (run `generate_comparisons.py` first). |
| `GET /embedding_3d` | Embeddings projected to 3D via SVD/PCA for visualization. |

## Getting started

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# (optional) retrain the model
python word2vec.py
python generate_comparisons.py

# serve the API
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dev server runs at `http://127.0.0.1:5173` and talks to the backend.

## Project layout

```
backend/
  bpe.py                    # BPE tokenizer
  word2vec.py               # skip-gram + negative sampling training
  main.py                   # FastAPI server
  generate_comparisons.py   # precompute comparison data
  corpus.txt                # training corpus
  model.npz / vocab.json    # trained artifacts
frontend/
  src/App.jsx               # 3D embedding explorer
```
