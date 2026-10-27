# RQ5 论文证据包

## 内容与单位

- figures/：5 张图，每张提供可编辑 SVG、矢量 PDF、300 dpi PNG。
- tables/：重新计算的数据汇总、逐 seed 配对差值、分腐蚀族/属性数据、训练摘要、MTL NME，以及论文 Markdown/LaTeX 表格。
- raw/：从 Drive 下载的原始 CSV、JSON 和九个新 runs、三个旧 A runs 的记录；未下载权重。
- source_manifest.json：每个下载文件的 Drive ID、来源链接、修改时间和 SHA256。
- audit_report.json：核对项目及验证边界。
- paper_findings_en.md / figure_captions.md：英文论文结论草稿和图注。
- scripts/build_evidence.py：可重建汇总与图表的分析脚本，不运行模型。

mAP/F1 原始数据范围为 0-1；主表以百分数显示。差值统一为百分点 pp。
先在每个 seed 内等权平均 20 个非 clean 条件，再计算三 seed 的 mean 和 sample SD (ddof=1)。
配对误差条采用差值本身的样本标准差，不能用模型各自标准差代替，也不是置信区间。
clean 与 corrupted 分开；不从条件数或属性数扩大独立样本量。

## 主要判断

N1 相对 N0 没有稳定总体收益。MTL 相对 A 的腐蚀 mAP 在三个 seed 中都下降，
冻结 F1 平均提高但并非三个 seed 一致，不能称为全面改善。按条件和属性的差异只作探索性解释。
新实验均已早停：N0/N1 为 4 轮，MTL 为 5、5、6 轮，新训练合计 40 轮。旧 A_seed0 实际训练 6 轮，而当前早停规则会在第 4 轮触发；两者均选择第 2 轮权重，应在协议差异中披露。

## 复现

python scripts/build_evidence.py

依赖 numpy、pandas、matplotlib、reportlab。图表为从原始汇总记录计算得到的静态科学图。
脚本会重新核对来源 hash、输出汇总表和五张图；PDF 综合报告由 report builder 另外生成。
原始资产未修改。当前论文尚未写入本轮结果。

## Git 归档补充

source_snapshot/ 为本次实际执行的八个源码文件，已与 Drive 冻结 SHA256 全部核对。
源码模板中 λ 为空是正常的；实际训练使用 raw/frozen_config.json 中的 1.89811。
本 Git 包不包含综合报告 PDF、Notebook、数据集或权重，Notebook 沿用你本地已下载的版本。
运行 scripts/verify_archive.py 可使用 Python 标准库核对归档。
