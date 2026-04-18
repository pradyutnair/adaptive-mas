#!/usr/bin/env python3
"""Compatibility retriever server for external baseline repos.

Serves the local 05-mas E5 sentence index with two response formats:
- Adaptive-RAG style: {"retrieval": [...]}
- Generic style: {"results": [[...]], "success": true}
"""

from __future__ import annotations

import argparse
import json
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, chunks_file: str, index_dir: str, model_name: str, corpus_name: str) -> None:
        self.corpus_name = corpus_name
        with open(Path(index_dir) / "sentence_index.pkl", "rb") as f:
            index = pickle.load(f)
        self.sentences = index["sentences"]
        self.embeddings = index["embeddings"]
        self.sentence_to_chunk = index["sentence_to_chunk"]
        self.chunks = index["chunks"]
        self.model = SentenceTransformer(model_name)
        with open(chunks_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.raw_chunks = raw

    def _query_embedding(self, query: str) -> np.ndarray:
        try:
            return self.model.encode([query], prompt_name="query", normalize_embeddings=True)[0]
        except TypeError:
            return self.model.encode([query], normalize_embeddings=True)[0]

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        q = self._query_embedding(query)
        sims = np.dot(self.embeddings, q)
        top_indices = np.argsort(sims)[::-1][: top_k * 3]
        best: dict[str, tuple[float, list[str]]] = {}
        for idx in top_indices:
            chunk_key = str(self.sentence_to_chunk[idx])
            sentence = self.sentences[idx]
            score = float(sims[idx])
            prev = best.get(chunk_key)
            if prev is None or score > prev[0]:
                best[chunk_key] = (score, [sentence])
            else:
                prev[1].append(sentence)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
        out = []
        for chunk_id, (score, matched) in ranked:
            chunk = self.chunks[int(chunk_id)] if isinstance(self.chunks, list) else self.chunks[int(chunk_id)]
            text = chunk["text"]
            out.append(
                {
                    "id": str(chunk_id),
                    "title": f"chunk_{chunk_id}",
                    "paragraph_text": text,
                    "text": text,
                    "score": score,
                    "corpus_name": self.corpus_name,
                    "matched_sentences": matched[:5],
                }
            )
        return out


def make_handler(retriever: Retriever):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self) -> None:
            if self.path in {"/health", "/retriever_info"}:
                self._json(200, {"success": True, "corpus_name": retriever.corpus_name})
                return
            self._json(404, {"success": False, "error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/retrieve":
                self._json(404, {"success": False, "error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
            if "query_text" in req:
                results = retriever.search(req["query_text"], int(req.get("max_hits_count") or 5))
                self._json(200, {"retrieval": results, "success": True})
                return
            queries = req.get("queries") or []
            topk = int(req.get("topk") or 5)
            self._json(200, {"results": [retriever.search(q, topk) for q in queries], "success": True})

        def log_message(self, *_args: Any) -> None:
            return

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-file", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    ap.add_argument("--corpus-name", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    retriever = Retriever(args.chunks_file, args.index_dir, args.embedding_model, args.corpus_name)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(retriever))
    server.serve_forever()


if __name__ == "__main__":
    main()
