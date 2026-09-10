# Fig. 5 分步诊断

入口：`analysis/diagnose_fig5_alignment.py`。仅创建分析输出，不训练、不保存修改后的模型，不覆盖原 checkpoint。
先同步当前分析脚本及 `src/`、`TSMNet/` 相关依赖。使用服务器原来的训练环境。

## 第一步：只检查完整模型的 21 号被试

在项目根目录运行：

```bash
python -u analysis/diagnose_fig5_alignment.py \
  --stage extract --models ms_tgc_spddsbn --subjects 21 \
  --output-root outputs/fig5_stew_v3 \
  --master-summary outputs/fig5_stew_v3/master_summary.csv \
  --device cpu --batch-size 16 \
  --output-dir results/diagnose_stew_s21
```

`--device cuda` 可用于加速前端。SPD 运算仍沿用模型自身的 CPU 路径。
该步骤使用所有源训练窗口、所有目标窗口评估；图和对齐指标使用固定平衡抽样。
源统计重估逐域进行，域内分批提取紧凑特征后一次拟合，不把全部时间特征堆入内存。
全局 SPDBN 和 Mean-CE 没有分域统计，作为不做干预的对照，标记 `not_applicable_no_domain_bn`。

## 第二步：用相同特征、相同样本比较降维

```bash
python -m pip install -r requirements-analysis.txt
python -u analysis/diagnose_fig5_alignment.py \
  --stage embed --output-dir results/diagnose_stew_s21 \
  --reducers umap_pca30,umap_no_pca,tsne
```

每个特征位置、每种降维得到一张四面板图：列为 saved/source_refit；
上排按类别着色，下排按被试着色；圆点为源，三角为目标。
上下排复用相同二维坐标；不同列独立拟合投影，不表示点的移动轨迹。
UMAP 仅在源样本拟合；t-SNE 联合源目标特征拟合且不使用标签。
`tsne` 对照不做 PCA 预降维，与旧图脚本默认先 PCA30 再 t-SNE 不同，元数据中明确记录。
类别颜色说明适用于默认二分类 STEW。

## 可选：扩展比较

前三个和完整模型一起：

```bash
python -u analysis/diagnose_fig5_alignment.py \
  --models mstgc_mean_ce,mstgc_dta_cheb_eudsbn,mstgc_dta_cheb_spdbn,ms_tgc_spddsbn \
  --subjects 21 --device cpu --batch-size 16 \
  --output-root outputs/fig5_stew_v3 \
  --master-summary outputs/fig5_stew_v3/master_summary.csv \
  --output-dir results/diagnose_stew_four_s21
```

TSMNet 用 `--models tsmnet` 和包含对应记录的输出目录/master summary。
确认单被试诊断后，`--subjects all` 遍历所选模型 summary 中全部被试。
多模型 all 各自遍历完成折，并不自动取交集；比较时应对齐 subject。
多个数据集请分别运行并指定不同输出目录。非默认采样率、自定义缓存或被试范围
需要先核对训练配置；本入口沿用图5的默认全数据集缓存重建规则。

## 输出及判读

输出子目录为 `<output-dir>/<model>/subject_21/`：

- `audit.json`：checkpoint SHA256、统计变更键、目标 logits 最大差异、预测一致性、训练记录。
- `classification.csv`：saved/source_refit 两条件下全量源、目标 accuracy/BAcc/F1。
- `alignment.csv`：固定平衡样本上的高维域差异与类别分离指标。
- `class_separation.csv`：源、目标各自内部的类别中心距离、类内方差、Fisher 比。
- `by_subject.csv`：每个被试的 BAcc 与类内/类间指标。
- `saved_predictions.csv`、`source_refit_predictions.csv`：每个窗口的标签、预测和 logits。
- `samples.csv`、`plot_manifest.csv`：完整样本及固定绘图抽样。
- `features.npz`：两个位置、两个条件的标准化特征，供离线复查。
- 第二步的 PNG、坐标 CSV 和降维 JSON。

两个位置为 representation（原图读出模块之前）及 classifier_input（最终线性分类层之前）。
MS-TGC 后者默认128维；TSMNet没有相同的128维投影，两个位置相同，不能伪称128维。
所有前后比较固定使用 saved 状态的源训练特征拟合的 StandardScaler。
这使统计重估对特征的影响不会被第二次标准化抵消；source_refit 的指标不等同于
重新运行原图脚本、重新拟合 scaler 得到的指标。

必须首先检查：权重及非源统计不变、目标 logits 在容差内一致、目标预测相同、checkpoint不变。
不满足时脚本报错；不要解释失败实验的改善。
若源准确率/逐被试分离改善且目标不变，支持源统计问题；是否普遍存在需再跑全部折。
若仅分类层输入的分离好，说明提取位置/指标影响解释。
若只换降维后散点改变，但 CSV 指标完全不变，说明降维影响显示。
任何情况下都不要求指标单调变好，不依据目标标签选择训练超参数。

同一个输出目录已有完成诊断或同名降维图片时会拒绝覆盖。重跑请使用新目录；
第一步失败后也建议新目录，保留失败的审计信息。

## 打包交回分析

完成前两步后：

```bash
python -c "import shutil; print(shutil.make_archive('diagnose_stew_s21', 'zip', 'results', 'diagnose_stew_s21'))"
```

请提供该 ZIP，若运行报错也提供完整终端日志。无需先重新训练或修改原训练结果。
