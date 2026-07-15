"""Step 0 - Ingest JSON documents from S3 into corpus/.

Downloads JSON or JSONL files from an S3 location and writes the text of
each document to corpus/<id>.txt, ready for the discovery pipeline.

Supported S3 layouts:
  - One .json file per document (object key becomes the document ID)
  - One or more .jsonl files (each line is a document; sequential IDs used)

The text field is extracted by a dot-separated path, so nested fields work:
  --text-field body
  --text-field article.content
  --text-field sections.0.text   (numeric path components index into lists)

Usage:
  python scripts/00_ingest_s3.py s3://my-bucket/path/to/docs/ \\
      --text-field body

  python scripts/00_ingest_s3.py s3://my-bucket/corpus.jsonl \\
      --text-field content --id-field doc_id --limit 200

Options:
  --text-field FIELD    Dot-separated path to the text field (default: text)
  --id-field FIELD      Field to use as filename stem; omit to use object key
  --limit N             Stop after N documents (for testing)
  --clear               Delete existing corpus/*.txt before ingesting
  --workers N           Parallel download threads (default: 8)

AWS credentials are read from the environment in the usual order:
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, IAM role.
"""

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

CORPUS = Path(__file__).parent.parent / "corpus"

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, prefix) from an s3:// URI."""
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    return p.netloc, p.path.lstrip("/")


def get_field(obj: dict, path: str):
    """Extract a value from a nested dict/list using a dot-separated path."""
    parts = path.split(".")
    cur = obj
    for part in parts:
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def safe_stem(s: str) -> str:
    """Turn any string into a safe filename stem (no slashes, spaces, etc.)."""
    s = re.sub(r"[^\w\-]", "_", s)
    return s[:120]  # cap length


def write_doc(text: str, stem: str) -> Path:
    dest = CORPUS / f"{stem}.txt"
    dest.write_text(text, encoding="utf-8")
    return dest


_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ── Single-object download ─────────────────────────────────────────────────────


def process_json_object(s3, bucket: str, key: str, text_field: str,
                        id_field: str | None, counter: list[int]) -> int:
    """Download one .json object, extract text, write to corpus/. Returns 1."""
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    obj = json.loads(body)
    text = str(get_field(obj, text_field))
    if not text.strip():
        return 0
    if id_field:
        stem = safe_stem(str(get_field(obj, id_field)))
    else:
        stem = safe_stem(Path(key).stem)
    write_doc(text, stem)
    n = counter[0] = counter[0] + 1
    if n % 100 == 0:
        log(f"  …{n} documents written")
    return 1


def process_jsonl_object(s3, bucket: str, key: str, text_field: str,
                         id_field: str | None, counter: list[int],
                         limit: int | None) -> int:
    """Download one .jsonl object, split by line, write each doc. Returns count."""
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    written = 0
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = str(get_field(obj, text_field))
        if not text.strip():
            continue
        n = counter[0] = counter[0] + 1
        if id_field:
            stem = safe_stem(str(get_field(obj, id_field)))
        else:
            stem = f"doc_{n:06d}"
        write_doc(text, stem)
        written += 1
        if n % 100 == 0:
            log(f"  …{n} documents written")
        if limit and n >= limit:
            break
    return written


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("s3_uri", help="s3://bucket/prefix/ or s3://bucket/file.jsonl")
    parser.add_argument("--text-field", default="text",
                        help="Dot-separated path to the text field (default: text)")
    parser.add_argument("--id-field", default=None,
                        help="Field to use as filename stem (default: S3 object key)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N documents")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing corpus/*.txt before ingesting")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads (default: 8)")
    args = parser.parse_args()

    if args.clear:
        removed = list(CORPUS.glob("*.txt"))
        for f in removed:
            f.unlink()
        log(f"Cleared {len(removed)} existing files from corpus/")

    bucket, prefix = parse_s3_uri(args.s3_uri)
    s3 = boto3.client("s3")

    # Detect whether the URI points to a single file or a prefix/folder.
    is_single_file = prefix.endswith(".json") or prefix.endswith(".jsonl")

    if is_single_file:
        objects = [prefix]
    else:
        log(f"Listing objects in s3://{bucket}/{prefix} …")
        paginator = s3.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".json") or key.endswith(".jsonl"):
                    objects.append(key)
        log(f"Found {len(objects)} objects")

    if not objects:
        log("No .json or .jsonl objects found. Check the S3 URI and your credentials.")
        sys.exit(1)

    counter = [0]  # mutable int shared across threads

    if any(k.endswith(".jsonl") for k in objects):
        # JSONL: process sequentially (lines must be counted globally for limit)
        for key in objects:
            log(f"Processing {key} …")
            process_jsonl_object(s3, bucket, key, args.text_field,
                                 args.id_field, counter, args.limit)
            if args.limit and counter[0] >= args.limit:
                break
    else:
        # Individual JSON files: download in parallel
        keys_to_fetch = objects[:args.limit] if args.limit else objects
        log(f"Downloading {len(keys_to_fetch)} documents with {args.workers} threads …")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_json_object, s3, bucket, key,
                            args.text_field, args.id_field, counter): key
                for key in keys_to_fetch
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    log(f"  ERROR on {futures[fut]}: {exc}")

    log(f"\nDone. {counter[0]} documents written to corpus/")
    log("Next: run ./run_pipeline.sh (or python scripts/01_characterize.py first)")


if __name__ == "__main__":
    main()
