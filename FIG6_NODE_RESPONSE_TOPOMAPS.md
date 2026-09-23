# 图6：学习到的图传播前后节点空间响应

新脚本：`analysis/fig6_node_response_topomaps.py`。不修改现有模型、图4、图5或旧图6脚本，不训练、不重估BN统计，不使用未训练权重生成论文结果。

## 精确定义

- 从实际完整AGMNet LOSO checkpoint加载权重，严格核对v3架构和目标适应记录。
- 每个目标被试使用其留出折的全部目标测试窗口；源训练数据仅用于重建训练时的输入标准化。
- 用graph模块的forward-pre-hook和forward-hook，在同一次完整eval前向中提取输入/输出，并获得分类logits用于重新计算目标BAcc。POST在通道可靠性加权和SPD构造之前。
- 当前网络真实形状是`[N,C,F,T]`，展平为`[N,C,F*T]`，对最后一维做L2。不是先时间平均后再取范数；两者不是相同量。脚本记录展平后维度，PRE/POST形状不一致时停止，避免维数变化造成隐式尺度差异。
- 每被试、每通道分别计算两类响应均值，输出High-Low。优先识别名含high的标签；否则默认最大标签减最小标签。EEGMAT等非Low/High名称会在报告中写明真实名称与整数标签，亦可通过配置`high_label`显式指定。
- 组图对被试等权平均，不按窗口数加权；不单独归一化PRE和POST。
- 图传播模块还包含可学习特征投影、偏置和ReLU，因此前后差异并非只由邻接传播单独造成。

## 布局与代表被试

采用最终布局要求的**4个代表被试+平均图**。原请求前文“5个代表被试”与最终“4+Avg”存在数量差异，以最终布局为准。

两行分别为STEW、EEGMAT。左侧每行：组平均PRE、组平均POST、POST符号一致性。右侧每行：Q25、中位数、Q75、均值附近各一位不重复的被试，以及Avg。

按重新计算的目标BAcc选择；重复时选下一个最接近且尚未使用者，距离相同时按subject_id升序。CSV记录选择准则、目标BAcc和实际BAcc。可固定给定四人，不按图形外观选人。

第三幅采用`mean(sign(delta_post))`：+1表示所有被试在该电极都是正对比，-1表示都是负对比；零对比贡献零。此图使用独立[-1,1]色条。它不是Pearson指标，也不表示统计显著性。所有响应对比图，包括代表被试，共享同一数据集的对称色限；可加`--color-scope global`使两个数据集也共用响应色限。默认范围覆盖该数据集所有被试PRE/POST值，避免只为选中个体缩放或裁掉极值。

图宽7.16英寸；PDF嵌入字体，PNG600dpi。使用MNE topomap；没有MNE时允许matplotlib线性插值，但必须提供真实/标准投影坐标CSV，不根据电极名称猜位置。插值区域外留白不能解释为零响应。

## 一致性指标与限制

分别对被试对的通道对比向量计算Pearson r，并取算术平均。常量图的Pearson相关性未定义，不填0；PRE/POST仅在共同有效的被试对上汇总。输出总对数、有效对数及每对相关性，便于审计。被试对不独立，不对这些pair做普通独立样本显著性检验。

不同被试对应不同LOSO模型，层响应绝对尺度可能因训练折而不同。共享色条仅保证显示尺度一致，不证明潜在表示已跨模型校准。一致性上升只能描述本次空间模式更一致，不自动证明泛化提升、时间稳定性或更集中。脚本不预设POST更好。

## 服务器运行

依赖沿用项目训练环境，额外需要MNE（优先绘图后端）；已在本地MNE1.7.1验证。若服务器未安装：

```bash
/root/miniconda3/bin/python -m pip install mne
```

先用现有STEW完整48折确认流程，输出独立目录，不覆盖旧图6：

```bash
cd ~/autodl-tmp/update-2.0/A-update
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/root/miniconda3/bin/python -u analysis/fig6_node_response_topomaps.py \
  --datasets stew --dataset-labels STEW \
  --output-root outputs/fig5_stew_v3 \
  --master-summary outputs/fig5_stew_v3/master_summary.csv \
  --data-root data --cache-root outputs/cache \
  --batch-size 16 --device cpu --threads 1 --seed 42 \
  --output-dir results/fig6_nodes_stew
```

生成要求的STEW+EEGMAT两行复合图：

1. 复制 `analysis/fig6_node_response_runs.example.json` 为自己的配置文件。
2. 把两个数据集的output_root、master_summary改成服务器已有完整AGMNet LOSO实验的真实路径。示例中的EEGMAT路径只是示例，不代表该训练已存在。
3. 运行：

```bash
/root/miniconda3/bin/python -u analysis/fig6_node_response_topomaps.py \
  --datasets stew,eegmat --dataset-labels STEW,EEGMAT \
  --run-config analysis/fig6_node_response_runs.example.json \
  --data-root data --cache-root outputs/cache \
  --batch-size 16 --device cpu --threads 1 --seed 42 \
  --output-dir results/fig6_nodes_stew_eegmat
```

若两个数据集都在同一输出根目录和master summary中，可不提供run-config，直接传`--output-root`与`--master-summary`。没有EEGMAT checkpoint时必须先完成该训练，不能用STEW或随机权重替代。

默认要求数据集里每个被试都有checkpoint及summary记录。调试子集可在配置中显式写`"subjects": [1, 5, 12, 21]`（至少四个真实被试），但该输出只是子集结果，不能写作全被试实验。

配置项示例（在对应数据集对象下增加）：

```json
{
  "representatives": [5, 12, 21, 34],
  "high_label": 1,
  "layout": "data/layouts/stew_scalp_xy.csv"
}
```

`representatives`必须是本次已评估的四个不同被试。`layout`为可选的真实头皮二维投影CSV，字段`channel,x,y`，x向右、y向前，坐标必须在相同单位中；脚本仅做统一显示比例变换。不提供时按标准10-20 montage定位。任何未知电极名、重复坐标或缺失坐标均报错，不静默替代。

## 输出和缓存

输出目录包含：

- `Fig6_node_response_topomaps.pdf` / `.png`
- `fig6_topomap_consistency.csv`
- `fig6_subject_selection.csv`
- `fig6_subject_contrasts.csv`：每被试每通道两类响应均值和差值
- `fig6_subject_bacc.csv`：重新评估BAcc、训练summary BAcc及差值
- `fig6_pairwise_correlations.csv`
- `fig6_summary.md` / `fig6_provenance.json`
- 各数据集的layout CSV
- `node_response_cache/<dataset>/subject_XX/`：响应、logits、标签、sample_id、subject_id、电极顺序与audit

缓存保存节点响应，不保存庞大的原始4维中间张量；真实中间特征仍由函数直接返回，并在同次前向中归约。每次提取检查权重/缓冲区不变、checkpoint SHA256不变，缓存同时核对数据cache哈希与输入配置。重跑同一配置会复用响应；改变checkpoint或提取配置时使用新目录。换代表被试或色条范围不需要重新提取。

若BAcc与训练summary不一致，请先看`bacc_delta_vs_summary`，核对CPU/GPU差异、checkpoint和数据版本，不根据差异较小自行忽略。代表选择以此次实际eval BAcc为准。

本地测试使用明确的数值与版式fixture，仅验证代码和图面；未生成真实科学结论。实际两数据集论文图需从服务器真实checkpoint执行以上命令后得到。
