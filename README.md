# Akkadian → English 翻译优化项目

本项目面向 Kaggle **Deep Past Challenge – Translate Akkadian to English**，目标是将带有变音符号、下标数字、破损标记和大量低频词形的阿卡德语音译文本翻译为英语。

本次修改依据项目根目录中的《Deep Past Challenge - Translate Akkadian to English 赛后总结20260324.pdf》，重点落实赛后方案反复验证的技术路线：

> 数据质量优先 → 可审计的句子级数据 → ByT5 字节级模型 → 无泄漏验证 → 多候选推理与可选 MBR → 轻量后处理。

需要特别说明：当前环境未下载或加载大型 ByT5 模型，也未执行 GPU 训练。因此本文中的性能结果分为“本地数据处理/代码验证结果”和“尚未实测的模型指标”，不会虚构 BLEU、chrF++ 或训练耗时。

## 目录结构

```text
.
├── deep-past-initiative-machine-translation/  # 官方 CSV、词典及测试数据
├── src/
│   ├── data_process.py                       # 官方数据清洗与规范化
│   ├── merge_datasets.py                     # 多源数据合并、去重、过滤
│   ├── train_sft_byt5.py                     # ByT5 训练与验证
│   └── infer_byt5.py                         # 批量推理、评估、可选 MBR
├── data/                                     # 运行数据处理后生成
│   ├── processed_train.json
│   └── final_sft_dataset.json
├── outputs/                                  # 训练和推理输出目录
├── requirements.txt
└── Deep Past Challenge - Translate Akkadian to English 赛后总结20260324.pdf
```

## 与赛后总结方案的对应关系

### 1. 数据工程

赛后总结指出，句子级高质量平行数据是最主要的得分来源。因此代码不再把原始 `train.csv` 直接拼接为指令数据，而是分为清洗、合并和训练划分三个阶段：

- `data_process.py` 保留 `source_raw` 和 `target_raw`，同时生成用于训练的规范化文本；
- 使用 Unicode NFC/NFKC、连续空白、破损文本标记和 Unicode 分数归一化；
- 使用 `<gap>` 统一常见缺失/破损标记；
- 每条样本保存 `oare_id`、来源文件、质量等级、原始行号和规范化配置；
- `merge_datasets.py` 兼容 `source/target` 与 `input/output` 两类历史 JSON 格式；
- 基于规范化后的源文本和目标文本精确去重；
- 通过目标/源字符长度比过滤明显异常样本；
- 使用固定随机种子，保证合并结果可复现。

原始文本始终保留，避免把有争议的音译规范化误当成唯一真值。赛后总结中对下标符号存在不同处理方案，本项目因此把规范化配置做成可切换参数，而不是硬编码一种不可回滚的规则。

### 2. 模型架构

训练脚本默认使用 `google/byt5-base`。ByT5 的字节级编码适合处理阿卡德语音译中的：

- `š`、`ṭ`、`ḫ` 等特殊字符；
- 下标数字和低频词形；
- 楔形文本转写中的破损标记和间隙；
- 人名、地名及其他低频专名。

当数据规模和显存允许时，可将 `--model-path` 替换为本地 ByT5-large 或 ByT5-XL。赛后总结中的大模型结果并不意味着 XL 应作为第一步配置；应先用 base 验证数据流程，再升级模型。

脚本支持可选 LoRA，但默认不启用。LoRA 适合显存受限的实验；如果目标是复现赛后总结中的标准 ByT5 主线，应先进行全量微调基线。

### 3. 训练和验证

- 任务固定为 `translate Akkadian to English: ...` 正向翻译；
- 按 `oare_id` 分组划分训练集和验证集，避免同一文档的句子泄漏到两侧；没有文档 ID 的样本才退化为单条样本分组；
- 使用动态 padding 和 `group_by_length=True`，减少无效 padding；
- 使用梯度累积实现较大的有效 batch；
- 默认关闭 FP16，避免赛后方案中提到的 NaN 稳定性问题；
- BF16 可通过 `--bf16 auto/on/off` 控制；
- 以 `eval_loss` 选择最佳检查点，避免在小验证集上过度依赖单一生成指标；
- 生成评估同时计算 BLEU、chrF++ 以及两者的几何平均：

\[
\text{score}=\sqrt{\text{BLEU}\times\text{chrF++}}
\]

### 4. 推理和 MBR

`infer_byt5.py` 提供：

- 批量输入和 beam search；
- 多候选生成；
- 基于候选之间 chrF++ 平均效用的轻量 MBR 选择；
- 相邻重复片段清理；
- 输入含参考译文时的 BLEU、chrF++ 和几何平均评估。

MBR 默认为可选功能。赛后总结显示，MBR 依赖候选多样性和数据质量，不能用来弥补脏数据；因此推荐先确认单模型收益，再启用多候选 MBR。后处理只做确定性的相邻重复清理和标点空白修正，不对人名、地名或词典结果进行强制字符串替换。

## 安装

建议使用 Python 3.10 或更高版本，并根据机器环境单独安装匹配的 PyTorch CPU/CUDA 版本。

```powershell
pip install -r requirements.txt
```

`requirements.txt` 已包含数据处理、Transformers、PEFT、数据集和 `sacrebleu` 评估依赖。PyTorch 未锁定版本，以避免覆盖已有的 CUDA 环境。

## 使用方法

以下命令均从项目根目录执行。

### 1. 清洗官方训练数据

```powershell
python src/data_process.py
```

默认读取：

```text
deep-past-initiative-machine-translation/train.csv
```

默认生成：

```text
data/processed_train.json
```

可选参数示例：

```powershell
python src/data_process.py `
  --unicode-form NFC `
  --output data/processed_train_nfkc.json
```

如果需要对比不同规范化方案，可使用 `--no-normalize-gaps` 或 `--no-normalize-fractions`。建议每次实验使用不同输出文件，并在固定验证集上做消融。

### 2. 合并和过滤数据

```powershell
python src/merge_datasets.py `
  data/processed_train.json `
  --output data/final_sft_dataset.json `
  --min-length-ratio 0.2 `
  --max-length-ratio 8.0 `
  --seed 42
```

可以继续传入多个外部 JSON 文件，例如 OCR 对齐数据、LLM 伪标签和人工校正数据。不同来源应在各自 JSON 的 `metadata.quality` 和 `metadata.source_dataset` 中保留来源信息。

### 3. 训练 ByT5

首先建议使用本地模型目录做小规模基线：

```powershell
python src/train_sft_byt5.py `
  --model-path path/to/byt5-base `
  --data-path data/final_sft_dataset.json `
  --output-dir outputs/byt5-base `
  --epochs 3 `
  --local-files-only
```

完整训练可调整：

```powershell
python src/train_sft_byt5.py `
  --model-path path/to/byt5-large `
  --data-path data/final_sft_dataset.json `
  --output-dir outputs/byt5-large `
  --max-source-length 768 `
  --max-target-length 384 `
  --batch-size 4 `
  --gradient-accumulation-steps 16 `
  --learning-rate 7e-5 `
  --num-beams 4 `
  --bf16 auto
```

显存不足时可启用 LoRA：

```powershell
python src/train_sft_byt5.py `
  --model-path path/to/byt5-base `
  --use-lora `
  --lora-r 16 `
  --lora-alpha 32
```

### 4. 批量推理和评估

```powershell
python src/infer_byt5.py `
  --model-path outputs/byt5-base/final `
  --input deep-past-initiative-machine-translation/test.csv `
  --output outputs/predictions.json `
  --batch-size 8 `
  --num-beams 4 `
  --local-files-only
```

启用多候选 MBR：

```powershell
python src/infer_byt5.py `
  --model-path outputs/byt5-base/final `
  --input deep-past-initiative-machine-translation/test.csv `
  --output outputs/predictions_mbr.json `
  --num-beams 8 `
  --num-candidates 4 `
  --mbr `
  --local-files-only
```

测试集没有参考译文时，脚本只输出预测结果，不会打印虚假的指标。使用带有 `translation` 列的验证 CSV 才能计算 BLEU、chrF++ 和几何平均。

## 已完成修改

### `src/data_process.py`

- 从模块级执行改为 `argparse + main()`；
- 修复相对路径依赖，默认基于项目根目录定位数据；
- 增加 Unicode、空白、gap、分数处理；
- 保留原始字段和来源元数据；
- 检查输入必要列，输出可复现 JSON。

### `src/merge_datasets.py`

- 支持多文件合并；
- 兼容旧版字段名；
- 精确去重；
- 长度比过滤；
- 固定种子打乱；
- 保存质量和来源元数据。

### `src/train_sft_byt5.py`

- 改为显式 CLI，不再导入即加载模型或启动训练；
- 按文档 ID 分组切分，降低验证泄漏；
- 使用 ByT5 正向 prompt；
- 动态 padding、长度分组、梯度累积；
- 默认 FP32，支持 BF16；
- 支持全量微调/LoRA；
- 计算 BLEU、chrF++ 和几何平均；
- 以 `eval_loss` 恢复最佳模型。

### `src/infer_byt5.py`

- 新增批量推理入口；
- 支持 beam search 和多候选；
- 可选 chrF++ MBR；
- 添加保守的相邻重复清理；
- 有参考译文时输出评估指标。

### `requirements.txt`

- 去除重复的 `jsonlines`；
- 补充 `pandas` 和 `sacrebleu`；
- 保留 PyTorch 由用户按硬件环境安装的策略。

## 性能评估

### 本地已验证结果

在未下载模型、不使用 GPU 的条件下完成：

| 项目 | 结果 |
|---|---:|
| Python 语法编译 | 4 个脚本全部通过 |
| CLI `--help` | 4 个脚本全部退出码 0 |
| 官方数据清洗 | 1,561 条输入，1,561 条非空输出 |
| 长度比过滤 | 38 条被过滤 |
| 合并后数据 | 1,523 条 |
| 去重统计 | 0 条重复 |
| 模型训练/推理 | 未执行 |

本地运行产生的 `processed_train.json` 和 `final_sft_dataset.json` 是基于当前官方 CSV 的可复现实验数据资产。上述数量不代表外部 PDF/OCR/LLM 数据已经加入训练。

### 尚未测量的指标

以下指标必须在实际加载模型和运行训练后填写，当前不做估计：

- 验证集 BLEU；
- 验证集 chrF++；
- BLEU/chrF++ 几何平均；
- GPU 显存和训练耗时；
- batch 推理吞吐量；
- MBR 相对单模型的增益；
- INT8/CTranslate2 相对原始模型的速度和质量变化。

正式实验时，建议按文档方案额外分桶报告短句/长句、gap、数字、分数、专有名词、OCR 来源和不同文档来源的指标，避免总体分数掩盖系统性错误。

## 推荐实验顺序

1. 使用官方清洗数据复现 ByT5-base 单模型；
2. 固定按文档分组的验证集，禁止相邻句跨集合；
3. 加入高置信 PDF/出版物句对，并保留来源质量标签；
4. 对 `published_texts.csv` 使用领域模型生成伪标签，再做过滤；
5. 在 base 有效后升级 large，数据充分时再尝试 XL；
6. 逐项消融规范化、持续预训练、双向训练和质量采样权重；
7. 确认单模型收益后启用多候选 MBR；
8. 最后评估 CTranslate2、INT8 和 GPU/CPU 并行；
9. 只保留在独立验证和分桶评估中稳定有效的轻量后处理。

## 当前限制

- 现有实现没有自动完成 PDF/OCR 版面分析；这需要外部 OCR/视觉模型和可追溯的页码、坐标元数据；
- 没有接入 OARE 或其他在线 LLM/API，避免训练脚本隐式访问外部服务；
- 当前数据处理默认使用官方 `train.csv`，外部出版物和未翻译语料需要按统一 JSON 格式另行导入；
- MBR 当前使用候选间 chrF++ 做无参考重排，不包含冠军方案中的 BLEU/Jaccard/长度综合打分；
- 尚未实现 CTranslate2 转换和 INT8 部署；
- 没有声称复现 PDF 中的竞赛 Private 分数。

## 未来改进方向

1. 建立带页码、坐标、OCR 版本、提示版本和质量分的出版物数据流水线；
2. 使用词典锚点、数字/专名一致性和单调词位置约束，自动验收 LLM 句子对齐；
3. 为人工数据、官方数据、OCR 数据和伪标签设置质量感知采样权重；
4. 研究阿卡德语单语持续预训练及其对句子级微调的影响；
5. 使用正字法多视图训练，比较原始、规范化和简化输入；
6. 以检索增强方式提供相似泥板、词典和出版物上下文，避免最终输出强制替换专名；
7. 加入 CTranslate2、INT8、批量并行和吞吐/质量对照实验；
8. 为亚述学研究者输出 Top-k 候选、置信度、低置信词段和数据出处。

## 参考

- 项目根目录：`Deep Past Challenge - Translate Akkadian to English 赛后总结20260324.pdf`
- Kaggle 数据目录：`deep-past-initiative-machine-translation/`
- 模型主线：ByT5 字节级序列到序列翻译
- 评估实现：SacreBLEU 的 BLEU 与 chrF++
