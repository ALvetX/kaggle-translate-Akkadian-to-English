import pandas as pd
import json
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
DEEPSEEK_API_KEY = ""
INPUT_CSV = "../deep-past-initiative-machine-translation/train.csv"
OUTPUT_JSON = "deepseek_aligned_dataset.json"
MAX_WORKERS = 10  # 并发线程数，根据你的 API 速率限制调整

# 初始化 DeepSeek 客户端 (完全兼容 OpenAI SDK)
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)


# ================= 清洗规则 =================
def is_valid_sentence(akkadian: str, english: str) -> bool:
    """本地规则：过滤掉没有训练价值的低质量残句"""
    if not akkadian or not english:
        return False

    # 规则 1: 英文翻译太短（少于 4 个词），比如 "Seal of PN", "One sheep"
    if len(english.split()) < 4:
        return False

    # 规则 2: 阿卡德语原文破损太严重（<gap> 或 x 超过一半）
    words = akkadian.split()
    if not words:
        return False
    gap_count = sum(1 for w in words if '<gap>' in w.lower() or w.lower() == 'x')
    if gap_count / len(words) > 0.4:  # 破损率超过 40% 直接丢弃
        return False

    # 规则 3: 英文包含太多不确定性标记
    if "(?)" in english or "[...]" in english:
        return False

    return True


# ================= 核心 LLM 处理逻辑 =================
def process_single_row(row):
    oare_id = row['oare_id']
    transliteration = str(row['transliteration']).strip()
    translation = str(row['translation']).strip()

    # 提示词设计：强调“丢弃残句”和“严格 JSON 格式”
    system_prompt = """
    You are an expert Assyriologist. Your task is to align paragraph-level Akkadian text with its English translation at the sentence level.

    CRITICAL INSTRUCTIONS:
    1. Split the English translation into independent, complete sentences.
    2. Match each English sentence to its exact corresponding Akkadian text segment.
    3. FILTERING: If a sentence is severely broken, lacks a clear subject/verb, or consists mostly of names/gaps, DO NOT include it.
    4. Return ONLY a valid JSON object with a "pairs" array.

    JSON FORMAT:
    {
      "pairs": [
        {"akkadian": "...", "english": "..."},
        {"akkadian": "...", "english": "..."}
      ]
    }
    """

    user_prompt = f"### Akkadian Text ###\n{transliteration}\n\n### English Translation ###\n{translation}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            response_format={"type": "json_object"},
            temperature=0.1,  # 保持极低的温度以保证提取的准确性
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        result_json = json.loads(response.choices[0].message.content)

        valid_pairs = []
        for pair in result_json.get('pairs', []):
            akk = pair.get('akkadian', '').strip()
            eng = pair.get('english', '').strip()

            # 本地二次清洗
            if is_valid_sentence(akk, eng):
                valid_pairs.append({
                    "akkadian": akk,
                    "english": eng
                })

        return {"oare_id": oare_id, "status": "success", "pairs": valid_pairs}

    except Exception as e:
        return {"oare_id": oare_id, "status": "error", "error_msg": str(e), "pairs": []}


# ================= 主控制流 =================
def main():
    print("正在加载数据...")
    df = pd.read_csv(INPUT_CSV)

    # 仅为了测试，你可以先切片 df.head(20) 跑一下看看效果
    # df = df.head(20)

    results = []
    total_extracted_sentences = 0

    print(f"开始使用 DeepSeek 并发处理 {len(df)} 条文档...")

    # 使用线程池并发请求 API
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_row = {executor.submit(process_single_row, row): row for _, row in df.iterrows()}

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(future_to_row), total=len(df), desc="Alignment Progress"):
            res = future.result()

            if res['status'] == 'success' and res['pairs']:
                for pair in res['pairs']:
                    results.append({
                        "oare_id": res['oare_id'],
                        "source": f"translate Akkadian to English: {pair['akkadian']}",
                        "target": pair['english']
                    })
                total_extracted_sentences += len(res['pairs'])

    # 保存为 ByT5 训练脚本可以直接读取的格式
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 40)
    print("🎉 处理完成！")
    print(f"原始段落数: {len(df)}")
    print(f"成功提取并清洗的高质量句子对: {total_extracted_sentences}")
    print(f"数据已保存至: {OUTPUT_JSON}")
    print("=" * 40)


if __name__ == "__main__":
    main()