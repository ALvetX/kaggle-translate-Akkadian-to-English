import json
import torch
import torch.nn.functional as F
import sacrebleu
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

# ================= 1. 配置区 =================
BASE_MODEL_PATH = "google/byt5-base"
SFT_LORA_PATH = "./byt5-akkadian-sft-lora/final_lora_weights"
DATA_PATH = "final_sft_dataset.json"
OUTPUT_DIR = "./byt5-akkadian-gspo-final"

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 384
GROUP_SIZE = 4
LEARNING_RATE = 5e-6
CLIP_EPSILON = 0.2  # GSPO 序列级裁剪的超参数

# ================= 2. 加载与格式化数据 =================
print("正在加载数据集，并格式化为 GSPO 所需结构...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

grpo_data = []
for item in data:
    prompt_text = f"{item['instruction']}\nInput: {item['input']}\nOutput:"
    grpo_data.append({
        'prompt': prompt_text,
        'answer': item['output']
    })

dataset = Dataset.from_list(grpo_data)


# ================= 3. 定义奖励函数 =================
def chrf_reward_func(prompts, completions, answer, **kwargs):
    chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
    rewards = []
    for comp, ref in zip(completions, answer):
        if not comp.strip():
            rewards.append(0.0)
            continue
        score = chrf_metric.sentence_score(comp, [ref]).score
        rewards.append(score / 100.0)
    return rewards


def penalty_reward_func(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        penalty = 0.0
        if comp.count("<gap>") > 3:
            penalty -= 0.5
        if "<<" in comp or ">>" in comp or "(?)" in comp:
            penalty -= 0.2
        if len(comp.split()) < 2:
            penalty -= 0.3
        rewards.append(penalty)
    return rewards


# ================= 4. 自定义 GSPOTrainer (完美适配 Seq2Seq) =================
class GSPOTrainer(GRPOTrainer):
    """
    继承 GRPOTrainer，重写核心的 Policy Loss 计算逻辑。
    不仅将其提升到序列级别 (GSPO)，同时适配 ByT5 的 Encoder-Decoder 传参规范。
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 直接获取 TRL 已经拆分好的 Prompt 和 Completion
        # 兼容不同 TRL 版本的键名
        prompt_ids = inputs.get("prompt_ids", inputs.get("prompt_input_ids"))
        prompt_mask = inputs.get("prompt_mask", inputs.get("prompt_attention_mask"))
        completion_ids = inputs.get("completion_ids")
        completion_mask = inputs.get("completion_mask")

        # 构建 labels：将 Decoder padding 的部分设置为 -100，让交叉熵忽略它们
        labels = completion_ids.clone()
        if completion_mask is not None:
            labels[completion_mask == 0] = -100
        else:
            labels[labels == model.config.pad_token_id] = -100

        # 2. 前向传播：完美的 Encoder-Decoder 传参方式
        outputs = model(input_ids=prompt_ids, attention_mask=prompt_mask, labels=labels)

        with torch.no_grad():
            # ================= 新增：兼容 LoRA (PeftModel) 的逻辑 =================
            if self.ref_model is None:
                # 如果是 PEFT 模型，TRL 不会创建 ref_model，通过禁用 adapter 获取参考输出
                with model.disable_adapter():
                    ref_outputs = model(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        labels=labels
                    )
            else:
                ref_outputs = self.ref_model(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    labels=labels
                )

        # 3. 提取 logits
        logits = outputs.logits
        ref_logits = ref_outputs.logits

        # 4. 计算逐 Token 的 log 概率 (排除 padding)
        loss_mask = (labels != -100)

        # clamp(min=0) 防御性编程，防止 -100 导致底层的 cross_entropy 报错
        logprobs = F.cross_entropy(logits.transpose(1, 2), labels.clamp(min=0), reduction="none")
        ref_logprobs = F.cross_entropy(ref_logits.transpose(1, 2), labels.clamp(min=0), reduction="none")

        # 将 padding 位置的概率强制置 0
        logprobs = logprobs * loss_mask
        ref_logprobs = ref_logprobs * loss_mask

        # 5. GSPO 核心：计算序列级别的聚合概率比
        seq_lengths = loss_mask.sum(dim=-1).float().clamp(min=1.0)

        seq_logprobs = logprobs.sum(dim=-1) / seq_lengths
        seq_ref_logprobs = ref_logprobs.sum(dim=-1) / seq_lengths

        # cross_entropy 返回的是负对数概率，因此这里用 ref - current
        ratio = torch.exp(seq_ref_logprobs - seq_logprobs)

        # 6. 获取 Group 内部计算好的 Advantages
        advantages = inputs.get("advantages", torch.zeros_like(ratio))

        # 7. GSPO 序列级裁剪
        clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)

        # 8. 计算最终的 Policy Loss
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 9. 计算 KL 散度惩罚
        kl_div = (logprobs - ref_logprobs).sum(dim=-1) / seq_lengths
        beta = getattr(self, 'beta', getattr(self.args, 'beta', 0.1))
        kl_loss = beta * kl_div.mean()

        loss = policy_loss + kl_loss

        return (loss, outputs) if return_outputs else loss


# ================= 5. 模型与分词器初始化 =================
# ================= 5. 模型与分词器初始化及劫持 =================
print("正在加载基础模型并合并 SFT LoRA 权重...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)

# [修复警告] 显式设置 Generation Config
base_model.generation_config.decoder_start_token_id = 0
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
base_model.generation_config.eos_token_id = tokenizer.eos_token_id

model = PeftModel.from_pretrained(base_model, SFT_LORA_PATH, is_trainable=True)

# [核心修复] 劫持 (Monkey Patch) ByT5 的 generate 方法
# 目的：将 Prompt 强行拼接到 ByT5 生成的回答前，防止 TRL 切片时切出空张量导致底层 C++ 崩溃
original_generate = model.generate


def patched_generate(*args, **kwargs):
    input_ids = kwargs.get("input_ids")
    if input_ids is None and len(args) > 0:
        input_ids = args[0]

    # 执行 ByT5 原本的生成逻辑 (只返回 completion)
    gen_outputs = original_generate(*args, **kwargs)

    # 强行拼接！伪装成 Causal LM 的输出格式 [prompt, completion]
    return torch.cat([input_ids, gen_outputs], dim=1)


model.generate = patched_generate

# ================= 6. GSPO 训练配置 =================
print("初始化 GSPOTrainer...")
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=GROUP_SIZE,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=100,
    report_to="none"
)

# 使用我们自定义的 GSPOTrainer 替换原有的 GRPOTrainer
trainer = GSPOTrainer(
    model=model,
    reward_funcs=[chrf_reward_func, penalty_reward_func],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ================= 7. 启动强化学习 =================
print("🚀 开始 GSPO 后训练...")
trainer.train()

trainer.save_model(f"{OUTPUT_DIR}/final_gspo_weights")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_gspo_weights")
print("🎉 GSPO 训练完成，终极权重已保存！")