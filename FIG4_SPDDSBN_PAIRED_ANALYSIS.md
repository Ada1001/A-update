# 图4：SPDDSBN 配对机制实验

入口：`analysis/fig_spddsbn_paired_mechanism_analysis.py`。仅完整模型 `ms_tgc_spddsbn`，不重新训练，不修改 checkpoint，不替换图5。

## 实验定义与解释边界

- PRE：**BiMap + ReEig 后、SPDDSBN 前**的 SPD 矩阵，默认 20×20。POST：SPDDSBN 后、LogEig 前的同维矩阵。不能把原始 65×65 AugSPD 和 20×20 输出直接作配对几何比较。
- 使用每折原 checkpoint。在内存模型上用全部源训练窗口重估各源域统计，保留目标域统计。审计只有源统计 buffer 改变、目标 logits 容差内一致且预测类别完全一致、checkpoint SHA256 不变。
- PRE/POST 来自同一次 eval forward。导出全部源训练窗口和目标测试窗口；验证集不进入分析。目标标签只用于事后类别分组与作图，不参与校准、公共参考、标准化和 PCA。
- 源 PRE 的 AIRM Karcher 均值作为公共参考；标准 sqrt(2) 非对角向量化；只在源 PRE 拟合 scaler 和 PCA。所有条件共用变换与坐标范围。均值迭代未收敛会报错，不能悄悄使用失败结果。
- 类条件域差异：各类别源/目标 AIRM 均值距离的等权平均。所有窗口参与均值。
- kNN：全部源窗口合池和全部目标窗口分别计算精确 AIRM 邻居，默认固定 k=15；不自动减小 k。源域此处包括源被试之间的关系，不等同于每个被试内部。POST 逐被试校准可能改变这些关系，需如实报告。
- Spearman：各组最多 50,000 个不重复、无序样本对；seed=42，同一对在 PRE/POST 中使用相同索引。
- Fisher：公共标准化高维切空间中 trace(SB)/(trace(SW)+1e-12)，同时输出 source/target/pooled，主图为 target。不是分类准确率，也不是最终 128 维分类特征。
- 默认48折，主图固定 subject21。指标报告该折百分位、距离中位数最近折、距离均值最近折；极端时警告，不自动换折。

这是**已训练网络内部、同一层前后的机制诊断**。PRE 已经过监督学习，不能称为“未经训练、未经分类的原始信号”。若要声称网络保留了原始输入的结构，仍需另做明确的输入表征基线；本图不能独自证明该命题。

## 服务器运行

先将新增脚本同步到服务器同一项目目录，使用此前成功运行图5的 Python 环境。以下路径沿用已有实验目录；如果最终论文 checkpoint 在另一目录，只修改 RUN_ROOT 与 MASTER。

```bash
cd ~/autodl-tmp/update-2.0/A-update
export RUN_ROOT=outputs/fig5_stew_v3
export MASTER="$RUN_ROOT/master_summary.csv"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
set -o pipefail
mkdir -p results/fig4_logs
```

先验证第21折完整流程（输出是调试用单折，不能当48折统计）：

```bash
/root/miniconda3/bin/python -u analysis/fig_spddsbn_paired_mechanism_analysis.py \
  --stage all --subjects 21 --expected-folds 1 \
  --representative-subject 21 \
  --output-root "$RUN_ROOT" --master-summary "$MASTER" \
  --data-root data --cache-root outputs/cache \
  --device cpu --batch-size 16 \
  --output-dir results/fig4_spddsbn_s21 \
  2>&1 | tee results/fig4_logs/s21.log
```

正式导出48折（与单折使用不同输出目录）：

```bash
/root/miniconda3/bin/python -u analysis/fig_spddsbn_paired_mechanism_analysis.py \
  --stage extract --subjects all --expected-folds 48 \
  --output-root "$RUN_ROOT" --master-summary "$MASTER" \
  --data-root data --cache-root outputs/cache \
  --device cpu --batch-size 16 \
  --output-dir results/fig4_spddsbn \
  2>&1 | tee results/fig4_logs/export.log
```

从已导出矩阵计算指标和出图：

```bash
/root/miniconda3/bin/python -u analysis/fig_spddsbn_paired_mechanism_analysis.py \
  --stage analyze --expected-folds 48 --representative-subject 21 \
  --k 15 --max-pairs 50000 --seed 42 --distance-block 256 \
  --output-dir results/fig4_spddsbn \
  2>&1 | tee results/fig4_logs/analyze.log
```

精确源域 kNN 的时间复杂度为 O(N_source²)，很可能比导出特征慢很多。50,000 只限制 Spearman 的采样对数，**不会减少 kNN 工作量**。逐行计算，不在内存中建立完整距离矩阵；每500行打印进度。`--distance-block` 控制临时矩阵批大小，不改变指标。这里不能用 UMAP 距离替代 AIRM 来提速。

分析阶段每完成一折保存带输入哈希和参数签名的结果。中断后重复 `--stage analyze` 命令会复用已审计完成的折；中断的当前折重新计算。改变 k、seed、均值参数等会重新计算。导出阶段不覆盖已有完整折；重复实验用新目录。不要把不同实验的导出文件手工拼接。

提取可以改 `--device cuda`，但为与已有 CPU 校准诊断保持一致建议先用 CPU；AIRM/均值/PCA 数值分析仍使用 CPU。不要混合设备结果作逐位数值一致性结论。

## 文件与六面板

结果目录内：

- `Fig_SPDDSBN_Paired_Mechanism.pdf`：7.16英寸双栏矢量图，嵌入字体。
- `Fig_SPDDSBN_Paired_Mechanism.png`：600 dpi。
- `spddsbn_paired_metrics_per_fold.csv`：每折样本数、域差异、邻居保持率、距离相关性、三种 Fisher。
- `spddsbn_paired_statistics.csv`：均值、标准差、中位数、IQR，配对差值、双侧 Wilcoxon p、秩二列效应量、中位配对差95% bootstrap区间。结构指标没有天然的 PRE/POST 两列，不伪造配对检验。
- `SPDDSBN_PAIRED_ANALYSIS_SUMMARY.md`：依据实际数据自动生成的方向、幅度、代表折位置和解释限制。
- `export_manifest.json`、`analysis_arguments.json`：折清单和参数。
- `folds/subject_XX/paired_spd.npz`：真实 PRE/POST、logits、sample_id、subject_id、fold_id、domain、true_label。
- 每折 `samples.csv`、`audit.json`、`common_coordinates.npz`、`metrics.json`、`analysis_complete.json`：配对身份、审计、公共变换及可复算结果。

(a,b) 同样本在共同 PCA 中的前后坐标；散点最多600个、按源/目标等额无标签抽样，仅限制显示数量。(c) 每个类别×域的投影算术中心箭头，空心为前、实心为后；这些中心不是用于指标的 Riemannian 均值。(d) 每折类条件 AIRM 前后配对。(e) 源/目标 kNN 和 Spearman 分布。(f) 目标 Fisher 前后配对。统计图显示全部折与中位数 bootstrap 95% CI。

理想支持证据是域差异下降、邻居与距离排序保留较充分、Fisher没有明显受损；结果不满足时仍原样输出。不能因为 Fisher 的 p>0.05 就声称维持不变，也不能仅凭视觉上的混合断言有效。LOSO 折共享训练被试，检验和 bootstrap应作为探索性汇总，不是独立重复实验的严格推断。

建议先返回单折日志与图检查流程，再返回全48折的两个CSV、summary和图。矩阵文件可能很大，初步解释不需要全部上传。

```bash
zip -r results/fig4_spddsbn_review.zip \
  results/fig4_spddsbn/Fig_SPDDSBN_Paired_Mechanism.pdf \
  results/fig4_spddsbn/Fig_SPDDSBN_Paired_Mechanism.png \
  results/fig4_spddsbn/spddsbn_paired_metrics_per_fold.csv \
  results/fig4_spddsbn/spddsbn_paired_statistics.csv \
  results/fig4_spddsbn/SPDDSBN_PAIRED_ANALYSIS_SUMMARY.md \
  results/fig4_spddsbn/export_manifest.json \
  results/fig4_spddsbn/analysis_arguments.json \
  results/fig4_spddsbn/folds/subject_*/audit.json \
  results/fig4_logs
```

本地测试仅在 pytest 临时目录用明确标注的数值测试数据验证 AIRM、邻居、Fisher、缓存防篡改、真实模型导出接口和图形布局；没有生成冒充真实实验的论文结果。
