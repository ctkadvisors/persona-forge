#!/usr/bin/env python3
"""Repair a GGUF whose header counts MTP layers the file does not contain.

Symptom: llama.cpp refuses the file with
    missing tensor 'blk.N.attn_norm.weight'
where N == the model's real layer count. Cause: the base config had
`mtp_num_hidden_layers` >= 1 (qwen3.5 family, dense models included), a PEFT
merge dropped the mtp.* tensors, and convert_hf_to_gguf.py (without --no-mtp)
still added them to block_count. Patching block_count alone is NOT enough —
llama.cpp then demands blk.N-1.nextn.* tensors instead. Both keys must change:

    {arch}.block_count           -= nextn_predict_layers
    {arch}.nextn_predict_layers   = 0

This edits the two u32 values in place (byte-for-byte same file size).
No dependencies. Usage:

    python3 scripts/fix_gguf_mtp.py model.gguf
"""
import struct
import sys

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode()


def skip_value(f, t):
    if t == 8:
        read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (cnt,) = struct.unpack("<Q", f.read(8))
        if et == 8:
            for _ in range(cnt):
                (n,) = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
        else:
            f.seek(SCALAR[et] * cnt, 1)
    else:
        f.seek(SCALAR[t], 1)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    with open(path, "r+b") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        f.read(4)  # version
        f.read(8)  # tensor count
        (n_kv,) = struct.unpack("<Q", f.read(8))

        offsets = {}  # suffix -> (offset, value)
        arch = None
        for _ in range(n_kv):
            key = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            if key == "general.architecture" and t == 8:
                arch = read_str(f)
                continue
            suffix = key.split(".", 1)[-1]
            if suffix in ("block_count", "nextn_predict_layers") and t == 4:
                off = f.tell()
                (val,) = struct.unpack("<I", f.read(4))
                offsets[suffix] = (off, val)
                continue
            skip_value(f, t)

        if "nextn_predict_layers" not in offsets:
            sys.exit(f"{path}: no nextn_predict_layers key — nothing to fix "
                     f"(arch={arch})")
        nextn_off, nextn = offsets["nextn_predict_layers"]
        if nextn == 0:
            print(f"{path}: nextn_predict_layers already 0 — nothing to do")
            return
        blocks_off, blocks = offsets["block_count"]

        f.seek(blocks_off)
        f.write(struct.pack("<I", blocks - nextn))
        f.seek(nextn_off)
        f.write(struct.pack("<I", 0))
    print(f"{path}: block_count {blocks} -> {blocks - nextn}, "
          f"nextn_predict_layers {nextn} -> 0 (arch={arch})")


if __name__ == "__main__":
    main()
