import argparse
import json
import math
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="批量 beam 推理，可选 chrF++ MBR，并在有参考译文时评估。")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "deep-past-initiative-machine-translation" / "test.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "predictions.json")
    parser.add_argument("--source-column", default="transliteration")
    parser.add_argument("--target-column", default="translation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--mbr", action="store_true", help="用候选间句级 chrF++ 平均效用选择译文")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def remove_adjacent_repetition(text):
    words = text.split()
    for width in range(min(12, len(words) // 2), 0, -1):
        index = 0
        cleaned = []
        while index < len(words):
            chunk = words[index:index + width]
            cleaned.extend(chunk)
            index += width
            while words[index:index + width] == chunk:
                index += width
        words = cleaned
    text = " ".join(words)
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def select_mbr(candidates, chrf):
    if len(candidates) == 1:
        return candidates[0]
    utilities = [
        sum(chrf.sentence_score(candidate, [other]).score for j, other in enumerate(candidates) if i != j)
        / (len(candidates) - 1)
        for i, candidate in enumerate(candidates)
    ]
    return candidates[max(range(len(candidates)), key=utilities.__getitem__)]


def main():
    args = parse_args()
    if args.mbr and args.num_candidates < 2:
        raise ValueError("--mbr 要求 --num-candidates 至少为 2")
    if args.num_candidates > args.num_beams:
        raise ValueError("--num-candidates 不能大于 --num-beams")

    import pandas as pd
    import torch
    from sacrebleu.metrics import BLEU, CHRF
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if args.input.suffix.lower() == ".csv":
        frame = pd.read_csv(args.input)
        records = frame.to_dict("records")
    else:
        with args.input.open("r", encoding="utf-8") as file:
            records = json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    chrf = CHRF(word_order=2)
    predictions = []

    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        prompts = [f"translate Akkadian to English: {row[args.source_column]}" for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_source_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_candidates,
                early_stopping=True,
            )
        decoded = [remove_adjacent_repetition(text) for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
        for offset, row in enumerate(batch):
            candidates = decoded[offset * args.num_candidates:(offset + 1) * args.num_candidates]
            prediction = select_mbr(candidates, chrf) if args.mbr else candidates[0]
            result = dict(row)
            result["prediction"] = prediction
            if args.num_candidates > 1:
                result["candidates"] = candidates
            predictions.append(result)

    references = [str(row.get(args.target_column, "")).strip() for row in predictions]
    hypotheses = [row["prediction"] for row in predictions]
    if references and all(references):
        bleu_score = BLEU().corpus_score(hypotheses, [references]).score
        chrf_score = chrf.corpus_score(hypotheses, [references]).score
        print(json.dumps({"BLEU": bleu_score, "chrF++": chrf_score, "geometric_mean": math.sqrt(bleu_score * chrf_score)}, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
    print(f"已输出 {len(predictions)} 条预测到 {args.output}")


if __name__ == "__main__":
    main()
