import os
import gc
import re
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType


# ==========================================
# 1. 基础配置 (Config)
# ==========================================
class Config:
    # 阿卡德语包含大量噪声和未知词，ByT5的字节级处理非常鲁棒
    MODEL_NAME = "../model/google/byt5-small"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 2e-4
    OUTPUT_DIR = "./byt5-akkadian-model"

    # LoRA 配置参数
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05


# 固定随机种子
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything()

# ==========================================
# 2. 数据加载与预处理
# ==========================================
INPUT_DIR = "../deep-past-initiative-machine-translation"
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")

# 加载词典和词汇表
lexicon_df = pd.read_csv(f"{INPUT_DIR}/OA_Lexicon_eBL.csv")
dictionary_df = pd.read_csv(f"{INPUT_DIR}/eBL_Dictionary.csv")

# 提取人名地名 (PN = Person Name, GN = Geographic Name)
proper_nouns = set(lexicon_df[lexicon_df['type'].isin(['PN', 'GN'])]['norm'].dropna())

# 构建阿卡德语->英语词典
akk_to_eng = dict(zip(dictionary_df['word'], dictionary_df['definition']))


def preprocess_text(text, mask_proper_nouns=True, replace_words=True):
    """预处理文本：mask人名地名，替换阿卡德语单词"""
    if pd.isna(text):
        return ""
    text = str(text)

    # Mask proper nouns
    if mask_proper_nouns:
        for noun in proper_nouns:
            text = re.sub(r'\b' + re.escape(noun) + r'\b', '[MASK]', text, flags=re.IGNORECASE)

    # Replace Akkadian words with English definitions
    if replace_words:
        words = text.split()
        replaced = []
        for word in words:
            clean_word = re.sub(r'[^\w-]', '', word)
            if clean_word in akk_to_eng:
                replaced.append(akk_to_eng[clean_word])
            else:
                replaced.append(word)
        text = ' '.join(replaced)

    return text


def simple_sentence_aligner(df):
    """将文档级别的对齐转换为句子级别的对齐"""
    aligned_data = []
    for idx, row in df.iterrows():
        src = preprocess_text(row['transliteration'])
        tgt = str(row['translation'])

        # 英文按标点断句，阿卡德文按换行断句
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]

        # 如果行数匹配，则假设是1对1的关系
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:  # 移除噪声数据
                    aligned_data.append({'transliteration': s, 'translation': t})
        else:
            # 匹配失败则保留原始文档对
            aligned_data.append({'transliteration': src, 'translation': tgt})

    return pd.DataFrame(aligned_data)


def create_bidirectional_data(dataset_split):
    """数据增强：双向互译 (Akk->Eng & Eng->Akk)"""
    df = dataset_split.to_pandas()

    # 方向1: Akkadian -> English
    df_fwd = df.copy()
    df_fwd['input_text'] = "translate Akkadian to English: " + df_fwd['transliteration'].astype(str)
    df_fwd['target_text'] = df_fwd['translation'].astype(str)

    # 方向2: English -> Akkadian
    df_bwd = df.copy()
    df_bwd['input_text'] = "translate English to Akkadian: " + df_bwd['translation'].astype(str)
    df_bwd['target_text'] = df_bwd['transliteration'].astype(str)

    # 拼接并打乱
    df_combined = pd.concat([df_fwd, df_bwd], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(df_combined)


# 执行数据对齐与划分
train_expanded = simple_sentence_aligner(train_df)
dataset = Dataset.from_pandas(train_expanded)
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)

# 执行双向数据增强
bidirectional_train = create_bidirectional_data(split_datasets['train'])
# 验证集通常保持单向（比赛只测 Akk -> Eng），这里为了演示流程对齐讲义逻辑
unidirectional_val = create_bidirectional_data(split_datasets['test'])

# ==========================================
# 3. 分词与模型输入化 (Tokenization)
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)


def preprocess_function(examples):
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["target_text"]]

    model_inputs = tokenizer(inputs, max_length=Config.MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=Config.MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_train = bidirectional_train.map(preprocess_function, batched=True)
tokenized_val = unidirectional_val.map(preprocess_function, batched=True)

# ==========================================
# 4. 模型加载与 LoRA 配置
# ==========================================
model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)

# 配置 LoRA
lora_config = LoraConfig(
    r=Config.LORA_R,
    lora_alpha=Config.LORA_ALPHA,
    lora_dropout=Config.LORA_DROPOUT,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 5. 训练参数设定 (Trainer)
# ==========================================
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=Config.LEARNING_RATE,
    optim="adafactor",
    label_smoothing_factor=0.2,
    # 关键修正：为了防止 NaN 错误，关闭 FP16
    fp16=False,
    # FP32 占用显存大，因此减小 batch size 并使用梯度累加
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    logging_steps=10,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("Starting Training...")
trainer.train()

# ==========================================
# 6. 保存模型
# ==========================================
trainer.save_model(Config.OUTPUT_DIR)
tokenizer.save_pretrained(Config.OUTPUT_DIR)
print(f"Model saved to {Config.OUTPUT_DIR}")