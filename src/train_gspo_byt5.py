import json
import torch
import torch.nn.functional as F
import sacrebleu
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

# ================= 1. 配置区 =================
BASE_MODEL_PATH = "google/byt5-small"
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


# ================= 4. 自定义 GSPOTrainer =================
class GSPOTrainer(GRPOTrainer):
    """
    继承 GRPOTrainer，重写核心的 Policy Loss 计算逻辑，
    将其从 Token 级别 (GRPO) 提升到序列级别 (GSPO)。
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 获取当前模型 (Policy) 和 参考模型 (Ref) 的输出
        outputs = model(**inputs)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)

        # 2. 提取 logits 和 labels
        logits = outputs.logits
        ref_logits = ref_outputs.logits
        labels = inputs["labels"]

        # 3. 计算逐 Token 的 log 概率
        # 注意: 忽略 padding token (label == -100)
        loss_mask = (labels != -100)

        # 计算当前模型的 Token 对数概率
        logprobs = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        # 计算参考模型的 Token 对数概率
        ref_logprobs = F.cross_entropy(ref_logits.transpose(1, 2), labels, reduction="none")

        # 4. GSPO 核心：计算序列级别的聚合概率比 (Sequence-level Likelihood Ratio)
        # 将一句话的所有 Token 概率求和，并除以句子有效长度进行归一化
        seq_lengths = loss_mask.sum(dim=-1).float()

        seq_logprobs = (logprobs * loss_mask).sum(dim=-1) / seq_lengths
        seq_ref_logprobs = (ref_logprobs * loss_mask).sum(dim=-1) / seq_lengths

        # 序列级重要性采样比率 r_i
        ratio = torch.exp(seq_ref_logprobs - seq_logprobs)  # 注意 cross_entropy 是负对数概率，所以顺序反转

        # 5. 获取 Group 内部计算好的 Advantages (由父类 GRPOTrainer 在前向传播时根据 Reward 计算得出)
        advantages = inputs.get("advantages", torch.zeros_like(ratio))

        # 6. GSPO 序列级裁剪 (Sequence-level Clipping)
        clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)

        # 7. 计算最终的 Policy Loss (取负值因为我们要最大化 Advantage)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 8. 计算 KL 散度惩罚 (保持模型不偏离预训练分布)
        kl_div = ((logprobs - ref_logprobs) * loss_mask).sum(dim=-1) / seq_lengths
        kl_loss = self.beta * kl_div.mean()  # beta 是 KL 惩罚系数

        loss = policy_loss + kl_loss

        return (loss, outputs) if return_outputs else loss


# ================= 5. 模型与分词器初始化 =================
print("正在加载基础模型并合并 SFT LoRA 权重...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
model = PeftModel.from_pretrained(base_model, SFT_LORA_PATH, is_trainable=True)

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