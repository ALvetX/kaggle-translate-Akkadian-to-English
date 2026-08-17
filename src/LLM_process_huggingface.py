import pandas as pd
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
DEEPSEEK_API_KEY = ""
INPUT_PARQUET = "../akkadian_english_sentences_alignment_2/data/train-00000-of-00001.parquet"
OUTPUT_JSON = "hf_deepseek_aligned_dataset.json"
MAX_WORKERS = 10  # 并发线程数

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)


# ================= 清洗规则 (保持严格标准) =================
def is_valid_sentence(akkadian: str, english: str) -> bool:
    """过滤没有训练价值的低质量残句"""
    if not akkadian or not english:
        return False

    # 规则 1: 英文太短的不要（无意义词组）
    if len(english.split()) < 4:
        return False

    # 规则 2: 阿卡德语原文破损太严重（<gap> 或 x 超过 40% 直接丢弃）
    words = akkadian.split()
    if not words:
        return False
    gap_count = sum(1 for w in words if '<gap>' in w.lower() or w.lower() == 'x')
    if gap_count / len(words) > 0.4:
        return False

    # 规则 3: 英文包含太多破损标记
    if "(?)" in english or "[...]" in english or "..." in english:
        return False

    return True


# ================= 核心 LLM 处理逻辑 =================
def process_single_row(row_id, transliteration, translation):
    """处理单行数据，交由 DeepSeek 拆分对齐"""
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
            temperature=0.1,
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

        return {"row_id": row_id, "status": "success", "pairs": valid_pairs}

    except Exception as e:
        return {"row_id": row_id, "status": "error", "error_msg": str(e), "pairs": []}


# ================= 主控制流 =================
def main():
    print("正在加载 Parquet 数据...")
    # 依赖 pyarrow 或 fastparquet
    df = pd.read_parquet(INPUT_PARQUET)

    # 【注意】HF 的数据集列名可能不一样，这里做一下兼容性检测
    # 如果列名是 'transliteration' 和 'translation'，则直接用；
    # 否则你需要根据 print 出来的列名修改下面的 get_text 函数。
    print(f"数据列名: {list(df.columns)}")

    def get_text(row):
        # 兼容不同命名习惯的 Hugging Face 数据集
        akk = row.get('transliteration') or row.get('akkadian') or row.get('source') or ""
        eng = row.get('translation') or row.get('english') or row.get('target') or ""

        # 处理部分 HF 数据集把翻译存成嵌套字典的情况，例如：{"en": "...", "akk": "..."}
        if isinstance(eng, dict) and 'en' in eng:
            akk = eng.get('akk', akk)
            eng = eng.get('en', eng)

        return str(akk), str(eng)

    results = []
    total_extracted_sentences = 0

    print(f"开始使用 DeepSeek 并发处理 {len(df)} 条文档...")

    # 并发请求 API
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 组装任务
        future_to_row = {}
        for idx, row in df.iterrows():
            akk_text, eng_text = get_text(row)
            if akk_text.strip() and eng_text.strip():
                future = executor.submit(process_single_row, idx, akk_text, eng_text)
                future_to_row[future] = idx

        # 进度条
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Alignment Progress"):
            res = future.result()

            if res['status'] == 'success' and res['pairs']:
                for pair in res['pairs']:
                    results.append({
                        "source": f"translate Akkadian to English: {pair['akkadian']}",
                        "target": pair['english']
                    })
                total_extracted_sentences += len(res['pairs'])

    # 保存结果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 40)
    print("🎉 处理完成！")
    print(f"原始段落数: {len(df)}")
    print(f"成功清洗出的高质量训练句子: {total_extracted_sentences}")
    print(f"数据已保存至: {OUTPUT_JSON}")
    print("=" * 40)


if __name__ == "__main__":
    main()