import pandas as pd
import json
from pathlib import Path

class AkkadianAligner:
    """阿卡德语句子对齐工具"""

    def __init__(self, data_dir='../deep-past-initiative-machine-translation'):
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.sentences_df = None
        self.lexicon_df = None
        self.published_df = None

    def load_data(self):
        """加载所有数据集"""
        print("加载数据...")
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.sentences_df = pd.read_csv(self.data_dir / 'Sentences_Oare_FirstWord_LinNum.csv')
        self.lexicon_df = pd.read_csv(self.data_dir / 'OA_Lexicon_eBL.csv')
        self.published_df = pd.read_csv(self.data_dir / 'published_texts.csv')

        print(f"训练集: {len(self.train_df)} 条")
        print(f"已发布文本: {len(self.published_df)} 条")
        print(f"句子分割: {len(self.sentences_df)} 条")
        print(f"词典: {len(self.lexicon_df)} 条")

    def align_train_data(self):
        """对齐训练数据的句子"""
        results = []
        lookup = self.lexicon_df.drop_duplicates(subset=['form']).set_index('form')[['norm', 'lexeme']].to_dict('index')

        for idx, row in self.train_df.iterrows():
            oare_id = row['oare_id']
            words = str(row['transliteration']).strip().split()
            doc_sents = self.sentences_df[self.sentences_df['text_uuid'] == oare_id].sort_values('first_word_obj_in_text')

            if len(doc_sents) == 0:
                continue

            chunks = []
            for sent_idx, (_, sent) in enumerate(doc_sents.iterrows()):
                start_idx = 0 if sent_idx == 0 else chunks[-1]['word_range'][1] + 1
                chunks.append({
                    "translation": str(sent['translation']),
                    "word_range": [start_idx, -1]
                })
                if sent_idx > 0:
                    chunks[-2]['word_range'][1] = start_idx - 1

            if chunks:
                chunks[-1]['word_range'][1] = len(words) - 1

            results.append({
                "oare_id": oare_id,
                "full_translation": str(row['translation']),
                "akkadian_words": [{"index": i, "word": w, "normalized": lookup.get(w, {}).get('norm', 'N/A')}
                                   for i, w in enumerate(words)],
                "aligned_chunks": chunks
            })

        return results

    def extend_from_published(self, max_samples=2000):
        """从published_texts扩展数据集（无翻译，仅用于预训练）"""
        results = []
        lookup = self.lexicon_df.drop_duplicates(subset=['form']).set_index('form')[['norm', 'lexeme']].to_dict('index')

        for _, row in self.published_df.head(max_samples).iterrows():
            oare_id = row['oare_id']
            words = str(row['transliteration']).strip().split()

            results.append({
                "oare_id": oare_id,
                "akkadian_words": [{"index": i, "word": w, "normalized": lookup.get(w, {}).get('norm', 'N/A')}
                                   for i, w in enumerate(words)],
                "source": "published_texts"
            })

        return results

    def save_results(self, aligned_data, extended_data, output_path='aligned_dataset.json'):
        """保存对齐结果"""
        output = {
            "train_aligned": aligned_data,
            "extended_unlabeled": extended_data,
            "stats": {
                "train_samples": len(aligned_data),
                "extended_samples": len(extended_data),
                "total": len(aligned_data) + len(extended_data)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n保存完成: {output_path}")
        print(f"训练对齐: {len(aligned_data)} 条")
        print(f"扩展数据: {len(extended_data)} 条")


if __name__ == "__main__":
    aligner = AkkadianAligner()
    aligner.load_data()

    print("\n对齐训练数据...")
    aligned = aligner.align_train_data()

    print("\n扩展数据集...")
    extended = aligner.extend_from_published(max_samples=2000)

    aligner.save_results(aligned, extended, 'enhanced_aligned_dataset.json')

