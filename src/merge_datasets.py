import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFIX = "translate Akkadian to English: "


def load_items(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} 的顶层必须是 JSON 数组")
    return data


def standardize(item, source_name):
    source = item.get("source", item.get("input", ""))
    target = item.get("target", item.get("output", ""))
    source = str(source).strip()
    if source.lower().startswith(PREFIX.lower()):
        source = source[len(PREFIX):].strip()
    target = str(target).strip()
    if not source or not target:
        return None
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("source_dataset", source_name)
    metadata.setdefault("quality", "unspecified")
    return {
        "instruction": "Translate the following Akkadian text to English.",
        "input": source,
        "output": target,
        "source_raw": item.get("source_raw", source),
        "target_raw": item.get("target_raw", target),
        "oare_id": str(item.get("oare_id", metadata.get("oare_id", ""))).strip(),
        "metadata": metadata,
    }


def merge_and_filter(file_paths, min_length_ratio, max_length_ratio, seed):
    unique_pairs = {}
    stats = {"loaded": 0, "empty": 0, "duplicate": 0, "length_ratio": 0}
    for path in file_paths:
        if not path.exists():
            print(f"警告: 找不到 {path}，已跳过")
            continue
        items = load_items(path)
        stats["loaded"] += len(items)
        for item in items:
            sample = standardize(item, str(path))
            if sample is None:
                stats["empty"] += 1
                continue
            ratio = len(sample["output"]) / max(len(sample["input"]), 1)
            if ratio < min_length_ratio or ratio > max_length_ratio:
                stats["length_ratio"] += 1
                continue
            key = (sample["input"], sample["output"])
            if key in unique_pairs:
                stats["duplicate"] += 1
                continue
            sample["metadata"]["target_source_char_ratio"] = round(ratio, 6)
            unique_pairs[key] = sample
    result = list(unique_pairs.values())
    random.Random(seed).shuffle(result)
    return result, stats


def parse_args():
    parser = argparse.ArgumentParser(description="合并平行数据，基于 source+target 去重并按字符长度比过滤。")
    parser.add_argument("inputs", nargs="*", type=Path, default=[PROJECT_ROOT / "data" / "processed_train.json"])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "final_sft_dataset.json")
    parser.add_argument("--min-length-ratio", type=float, default=0.2)
    parser.add_argument("--max-length-ratio", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_length_ratio <= 0 or args.max_length_ratio <= args.min_length_ratio:
        raise ValueError("长度比阈值无效")
    samples, stats = merge_and_filter(args.inputs, args.min_length_ratio, args.max_length_ratio, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
    print(f"已输出 {len(samples)} 条到 {args.output}; 统计: {stats}")


if __name__ == "__main__":
    main()
