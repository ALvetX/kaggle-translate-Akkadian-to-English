import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ================= 配置区 =================
BASE_MODEL_PATH = "google/byt5-small"
FINAL_LORA_PATH = "./byt5-akkadian-gspo-final/final_gspo_weights"  # 你的最终 GSPO 权重路径
MERGED_OUTPUT_PATH = "./byt5-akkadian-ultimate-merged"  # 合并后的标准模型输出路径


def merge_and_save():
    print(f"1. 正在加载基础模型: {BASE_MODEL_PATH}")
    # 建议在 CPU 上合并以避免显存不足，或者如果你显存够大，也可以 .to("cuda")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    print(f"2. 正在加载最终的 LoRA 适配器: {FINAL_LORA_PATH}")
    # 将 LoRA 挂载到基础模型上
    model = PeftModel.from_pretrained(base_model, FINAL_LORA_PATH)

    print("3. 正在执行物理权重合并 (Merge and Unload)...")
    # 核心操作：将 LoRA 矩阵 (A * B) 物理加到原线性层的权重 (W) 上，并卸载 Peft 结构
    merged_model = model.merge_and_unload()

    print(f"4. 正在保存合并后的标准大模型到: {MERGED_OUTPUT_PATH}")
    # 保存为标准的 Hugging Face 模型格式
    merged_model.save_pretrained(MERGED_OUTPUT_PATH)
    tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

    print("🎉 合并大功告成！")
    print("现在你可以像加载普通预训练模型一样，直接使用 AutoModelForSeq2SeqLM 加载它了！")


if __name__ == "__main__":
    merge_and_save()