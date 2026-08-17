import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "deep-past-initiative-machine-translation"
FRACTIONS = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5", "⅙": "1/6",
    "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}
GAP_RE = re.compile(r"(?<!\w)(?:\.{2,}|…+|x{2,}|X{2,}|\[?x+\]?)(?!\w)")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text, unicode_form="NFC", normalize_gaps=True, normalize_fractions=True):
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize(unicode_form, text)
    if normalize_fractions:
        text = "".join(FRACTIONS.get(char, char) for char in text)
        text = re.sub(r"(?<!\d)(\d+)\s*[⁄∕]\s*(\d+)(?!\d)", r"\1/\2", text)
    if normalize_gaps:
        text = GAP_RE.sub("<gap>", text)
    return SPACE_RE.sub(" ", text).strip()


def build_samples(train_df, unicode_form="NFC", normalize_gaps=True, normalize_fractions=True):
    samples = []
    for row_number, row in train_df.iterrows():
        source_raw = "" if pd.isna(row.get("transliteration")) else str(row["transliteration"])
        target_raw = "" if pd.isna(row.get("translation")) else str(row["translation"])
        source = normalize_text(source_raw, unicode_form, normalize_gaps, normalize_fractions)
        target = normalize_text(target_raw, unicode_form, False, normalize_fractions)
        if not source or not target:
            continue
        oare_id = str(row.get("oare_id", "")).strip()
        samples.append({
            "source": source,
            "target": target,
            "source_raw": source_raw,
            "target_raw": target_raw,
            "oare_id": oare_id,
            "metadata": {
                "source_dataset": "deep-past-initiative-machine-translation/train.csv",
                "quality": "official_document_translation",
                "row_number": int(row_number),
                "normalization": {
                    "unicode_form": unicode_form,
                    "whitespace": True,
                    "gaps": normalize_gaps,
                    "fractions": normalize_fractions,
                },
            },
        })
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="清洗官方阿卡德语-英语平行数据，同时保留原始文本和来源元数据。")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "train.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed_train.json")
    parser.add_argument("--unicode-form", choices=["NFC", "NFKC"], default="NFC")
    parser.add_argument("--no-normalize-gaps", action="store_true")
    parser.add_argument("--no-normalize-fractions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_csv(args.input)
    required = {"transliteration", "translation"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"输入缺少必要列: {sorted(missing)}")
    samples = build_samples(
        train_df,
        unicode_form=args.unicode_form,
        normalize_gaps=not args.no_normalize_gaps,
        normalize_fractions=not args.no_normalize_fractions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
    print(f"已输出 {len(samples)} 条样本到 {args.output}")


if __name__ == "__main__":
    main()
