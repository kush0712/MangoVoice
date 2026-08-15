"""
MangoVoice — Build LanceDB index from raw passages JSONL.

Usage:
  python -m ingestion.build_index \
      --input data/raw_passages.jsonl \
      --output data/lancedb \
      --strategy parent_child \
      --table chunks

Embeds all chunks with FastEmbed, stores in LanceDB with BM25 FTS enabled.
Writes manifest.json on completion.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def build_index(
    input_path: str,
    output_path: str,
    strategy: str = "parent_child",
    table_name: str = "chunks",
    batch_size: int = 512,
    max_passages: int = 5000,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> dict:
    """
    Build a LanceDB index from a JSONL passages file.
    Returns manifest stats.
    """
    import lancedb
    import pyarrow as pa
    from fastembed import TextEmbedding
    from ingestion.chunkers import chunk_passage

    print(f"Loading embedding model: {embedding_model}")
    embedder = TextEmbedding(model_name=embedding_model, cache_dir="data/fastembed_cache")

    # Wrap embedder for chunkers that need it
    class EmbedderWrapper:
        def embed(self, texts): return np.array(list(embedder.embed(texts)), dtype=np.float32)

    embedder_wrap = EmbedderWrapper()

    print(f"Reading passages from {input_path} (limit={max_passages})...")
    passages = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
            if max_passages and len(passages) >= max_passages:
                break

    print(f"Total passages loaded: {len(passages)}")

    # Chunk all passages
    print(f"Chunking with strategy: {strategy}")
    all_chunks = []
    for passage in passages:
        text = passage.get("text", "")
        parent_id = passage.get("passage_id", "")
        query_id = passage.get("query_id", "")
        language = passage.get("language", "en")

        chunks = chunk_passage(
            passage=text,
            parent_id=parent_id,
            query_id=query_id,
            language=language,
            strategy=strategy,
            embedder=embedder_wrap if strategy == "semantic" else None,
        )
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # Embed in batches
    print("Embedding chunks...")
    texts = [c.text for c in all_chunks]
    vectors = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vecs = list(embedder.embed(batch))
        vectors.extend(batch_vecs)
        if (i // batch_size + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Embedded {i + len(batch)}/{len(texts)} chunks ({elapsed:.1f}s)")

    print(f"Embedding done in {time.perf_counter() - t0:.1f}s")

    # Build LanceDB table
    print(f"Building LanceDB table at {output_path}/{table_name}...")
    db = lancedb.connect(output_path)

    # Build schema
    schema = pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("parent_id", pa.string()),
        pa.field("query_id", pa.string()),
        pa.field("language", pa.string()),
        pa.field("strategy", pa.string()),
        pa.field("chunk_start", pa.int32()),
        pa.field("chunk_end", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)),
    ])

    rows = [
        {
            "chunk_id": c.chunk_id,
            "parent_id": c.parent_id,
            "query_id": c.query_id,
            "language": c.language,
            "strategy": c.strategy,
            "chunk_start": c.chunk_start,
            "chunk_end": c.chunk_end,
            "text": c.text,
            "vector": vec.tolist() if hasattr(vec, "tolist") else list(vec),
        }
        for c, vec in zip(all_chunks, vectors)
    ]

    table = db.create_table(table_name, data=rows, schema=schema, mode="overwrite")

    # Build ANN index (only if sufficient rows for IVF_PQ)
    if len(rows) >= 256:
        print("Building ANN index (IVF_PQ)...")
        table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_PQ",
            num_partitions=min(256, max(1, len(rows) // 1000)),
        )
    else:
        print(f"Skipping IVF_PQ indexing ({len(rows)} < 256 rows) — using exact vector search.")

    # Build BM25 FTS index
    print("Building BM25 FTS index...")
    table.create_fts_index("text", replace=True)

    row_count = table.count_rows()
    print(f"\nIndex built: {row_count} rows in {output_path}/{table_name}")

    manifest = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "embedding_model": embedding_model,
        "chunking_strategy": strategy,
        "index_path": output_path,
        "table_name": table_name,
        "total_passages": len(passages),
        "total_chunks": len(all_chunks),
        "indexed_rows": row_count,
        "vector_dim": 384,
        "index_type": "IVF_PQ + BM25",
    }

    manifest_path = Path(output_path).parent / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MangoVoice LanceDB index")
    parser.add_argument("--input", default="data/raw_passages.jsonl")
    parser.add_argument("--output", default="data/lancedb")
    parser.add_argument("--strategy", default="parent_child",
                        choices=["canonical", "sentence_window", "fixed_token", "semantic", "parent_child"])
    parser.add_argument("--table", default="chunks")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-passages", type=int, default=5000)
    args = parser.parse_args()

    manifest = build_index(
        input_path=args.input,
        output_path=args.output,
        strategy=args.strategy,
        table_name=args.table,
        batch_size=args.batch_size,
        max_passages=args.max_passages,
    )
    print("\nManifest:")
    print(json.dumps(manifest, indent=2))
