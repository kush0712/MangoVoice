"""
MangoVoice — Stream & sample the MSMARCO-XI dataset.

Usage:
  python -m ingestion.stream_dataset --output data/raw_passages.jsonl \
         --max-rows 15000 --languages hi en --seed 42

Streams the training split, deduplicates passages, writes to JSONL.
Never loads the full 55.6 GB dataset into memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def hash_passage(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def stream_dataset(
    output_path: str,
    val_output_path: str = "data/val_passages.jsonl",
    max_rows: int = 3000,
    max_val_rows: int = 500,
    seed: int = 42,
) -> dict:
    """
    Stream MSMARCO-XI from Hugging Face parquet, deduplicate passages, write to JSONL.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    random.seed(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_output_path = Path(val_output_path)
    val_output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    passage_count = 0

    print("Downloading MSMARCO-XI Hindi parquet (validation/hinval.parquet)...")
    val_file = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        repo_type="dataset",
        filename="validation/hinval.parquet",
    )

    print(f"Reading row group from {val_file}...")
    pf = pq.ParquetFile(val_file)
    table = pf.read_row_group(0)
    total_available = len(table)
    print(f"Total available rows: {total_available}. Ingesting {max_rows} training rows + {max_val_rows} validation rows...")

    # 1. Write training passages (rows 0 to max_rows)
    print(f"Writing training passages to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for row_idx in range(min(max_rows, total_available)):
            row = table.slice(row_idx, 1).to_pylist()[0]
            query_id = str(row.get("query_id", row_idx))
            hi_query = str(row.get("query", "") or "").strip()
            en_query = str(row.get("Eng_Query", "") or "").strip()
            passages = row.get("passages") or {}

            hi_passages = passages.get("Translated_passages", []) or []
            en_passages = passages.get("English_passages", []) or []
            is_selected = passages.get("is_selected", []) or [0] * max(len(hi_passages), len(en_passages))

            for i, p_hi in enumerate(hi_passages):
                p_hi = str(p_hi).strip()
                if p_hi and len(p_hi) >= 20:
                    h = hash_passage(p_hi)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        sel = int(is_selected[i]) if i < len(is_selected) else 0
                        rec = {
                            "passage_id": f"hi-{query_id}-{i}",
                            "query_id": query_id,
                            "language": "hi",
                            "text": p_hi,
                            "is_selected": sel,
                            "query": hi_query,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        passage_count += 1

            for i, p_en in enumerate(en_passages):
                p_en = str(p_en).strip()
                if p_en and len(p_en) >= 20:
                    h = hash_passage(p_en)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        sel = int(is_selected[i]) if i < len(is_selected) else 0
                        rec = {
                            "passage_id": f"en-{query_id}-{i}",
                            "query_id": query_id,
                            "language": "en",
                            "text": p_en,
                            "is_selected": sel,
                            "query": en_query,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        passage_count += 1

            if (row_idx + 1) % 500 == 0:
                print(f"  Processed {row_idx + 1}/{max_rows} rows, {passage_count} unique passages...")

    print(f"Finished raw passages: {passage_count} unique passages written to {output_path}")

    # 2. Write validation queries (disjoint slice: rows max_rows to max_rows + max_val_rows)
    val_start = max_rows
    val_end = min(val_start + max_val_rows, total_available)
    val_rows_written = 0

    print(f"Writing {max_val_rows} disjoint validation queries to {val_output_path}...")
    with open(val_output_path, "w", encoding="utf-8") as vf:
        for val_idx in range(val_start, val_end):
            vrow = table.slice(val_idx, 1).to_pylist()[0]
            v_qid = str(vrow.get("query_id", val_idx))
            v_hi_query = str(vrow.get("query", "") or "").strip()
            v_en_query = str(vrow.get("Eng_Query", "") or "").strip()
            v_passages = vrow.get("passages") or {}

            v_hi_texts = v_passages.get("Translated_passages", []) or []
            v_en_texts = v_passages.get("English_passages", []) or []
            v_is_selected = v_passages.get("is_selected", []) or []

            gold_parent_ids = []
            for j, sel in enumerate(v_is_selected):
                if sel == 1:
                    gold_parent_ids.append(f"hi-{v_qid}-{j}")
                    gold_parent_ids.append(f"en-{v_qid}-{j}")

            # Alternate Hindi and English queries
            q_text = v_hi_query if (val_idx - val_start) % 2 == 0 else v_en_query
            q_lang = "hi" if (val_idx - val_start) % 2 == 0 else "en"

            if q_text:
                val_record = {
                    "query_id": v_qid,
                    "query": q_text,
                    "language": q_lang,
                    "gold_parent_ids": gold_parent_ids,
                    "is_selected_list": v_is_selected,
                }
                vf.write(json.dumps(val_record, ensure_ascii=False) + "\n")
                val_rows_written += 1

    print(f"Validation queries ready: {val_rows_written} queries in {val_output_path}")

    return {
        "train_query_rows": max_rows,
        "unique_passages": passage_count,
        "validation_queries": val_rows_written,
        "raw_passages_file": str(output_path),
        "val_passages_file": str(val_output_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream MSMARCO-XI dataset")
    parser.add_argument("--output", default="data/raw_passages.jsonl")
    parser.add_argument("--val-output", default="data/val_passages.jsonl")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-val-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = stream_dataset(
        output_path=args.output,
        val_output_path=args.val_output,
        max_rows=args.max_rows,
        max_val_rows=args.max_val_rows,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))

