# preprocess.py
# Converts CSVs (opening, closing) into fixed-length token id arrays (.npy)

import os
import json
import numpy as np
import pandas as pd

# ---------- Config ----------
RAW_DIR = "raw_data"
OUT_DIR = "processed_data"
FILES = ["test_34.csv"]
MAX_LEN = 71
# ---------------------------

def load_vocab(path):
    with open(path, "r") as f:
        tok2id = json.load(f)
    required = ["BOS", "SEP", "EOS", "PAD", "(", "[", ")", "]"]
    missing = [k for k in required if k not in tok2id]
    if missing:
        raise ValueError(f"Missing tokens in vocab: {missing}")
    return tok2id

def encode_sequence(opening, closing, tok2id):
    # Map chars to ids
    try:
        inp_ids = [tok2id[c] for c in opening]
        tgt_ids = [tok2id[c] for c in closing]
    except KeyError as e:
        raise ValueError(f"Unknown token in data: {e}")

    seq = (
        [tok2id["BOS"]]
        + inp_ids
        + [tok2id["SEP"]]
        + tgt_ids
        + [tok2id["EOS"]]
    )

    if len(seq) > MAX_LEN:
        raise ValueError(
            f"Sequence length {len(seq)} exceeds MAX_LEN={MAX_LEN}. "
            f"Opening len={len(opening)}, closing len={len(closing)}"
        )

    # Pad AFTER EOS
    pad_len = MAX_LEN - len(seq)
    if pad_len > 0:
        seq = seq + [tok2id["PAD"]] * pad_len

    return seq

def process_file(in_path, out_path, tok2id):
    df = pd.read_csv(in_path)
    if "opening" not in df.columns or "closing" not in df.columns:
        raise ValueError(f"{in_path} must have 'opening' and 'closing' columns")

    seqs = np.zeros((len(df), MAX_LEN), dtype=np.int16)

    for i, row in df.iterrows():
        opening = str(row["opening"])
        closing = str(row["closing"])
        seqs[i] = encode_sequence(opening, closing, tok2id)

    np.save(out_path, seqs)
    print(f"Saved: {out_path}  shape={seqs.shape}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    vocab_path = os.path.join(RAW_DIR, "token_to_id.json")
    tok2id = load_vocab(vocab_path)

    for fname in FILES:
        in_path = os.path.join(RAW_DIR, fname)
        out_name = os.path.splitext(fname)[0] + ".npy"
        out_path = os.path.join(OUT_DIR, out_name)
        process_file(in_path, out_path, tok2id)

if __name__ == "__main__":
    main()

