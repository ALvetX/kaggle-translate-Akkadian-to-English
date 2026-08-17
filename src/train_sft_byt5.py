import argparse
import json
import math
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="按文档无泄漏划分并微调 ByT5 进行阿卡德语到英语翻译。")
    parser.add_argument("--model-path", default="google/byt5-base", help="本地模型目录或 Hugging Face 模型名")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data" / "final_sft_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "byt5-akkadian")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--bf16", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--local-files-only", action="store_true", help="禁止模型加载访问网络")
    return parser.parse_args()


def load_and_split(path, eval_ratio, seed):
    with path.open("r", encoding="utf-8") as file:
        items = json.load(file)
    if len(items) < 2:
        raise ValueError("至少需要两条样本才能划分训练集和验证集")
    groups = {}
    for index, item in enumerate(items):
        group = str(item.get("oare_id") or item.get("metadata", {}).get("oare_id") or f"__row_{index}")
        groups.setdefault(group, []).append(item)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    eval_group_count = min(max(1, round(len(group_ids) * eval_ratio)), len(group_ids) - 1)
    eval_groups = set(group_ids[:eval_group_count])

    def format_item(item):
        source = item.get("input", item.get("source", "")).strip()
        target = item.get("output", item.get("target", "")).strip()
        return {"source": f"translate Akkadian to English: {source}", "target": target}

    train = [format_item(item) for group, rows in groups.items() if group not in eval_groups for item in rows]
    evaluation = [format_item(item) for group, rows in groups.items() if group in eval_groups for item in rows]
    return train, evaluation


def main():
    args = parse_args()
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval-ratio 必须在 0 和 1 之间")

    import numpy as np
    import torch
    from datasets import Dataset
    from sacrebleu.metrics import BLEU, CHRF
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    train_rows, eval_rows = load_and_split(args.data_path, args.eval_ratio, args.seed)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=args.local_files_only)

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        ))
        model.print_trainable_parameters()

    def preprocess(examples):
        inputs = tokenizer(examples["source"], max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=examples["target"], max_length=args.max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["source", "target"])
    eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=["source", "target"])
    bleu = BLEU()
    chrf = CHRF(word_order=2)

    def compute_metrics(prediction):
        predictions, labels = prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        hypotheses = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = bleu.corpus_score(hypotheses, [references]).score
        chrf_score = chrf.corpus_score(hypotheses, [references]).score
        return {"bleu": bleu_score, "chrf++": chrf_score, "bleu_chrf_geomean": math.sqrt(bleu_score * chrf_score)}

    use_bf16 = args.bf16 == "on" or (args.bf16 == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=3,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_target_length,
        bf16=use_bf16,
        fp16=False,
        group_by_length=True,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        report_to="none",
        seed=args.seed,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    print(f"训练样本: {len(train_dataset)}，验证样本: {len(eval_dataset)}，BF16: {use_bf16}")
    trainer.train()
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
