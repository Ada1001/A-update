# 图6排版与探索性统计 v2

上传以下两个脚本到服务器同名位置：

- `analysis/fig6_node_response_topomaps.py`
- `analysis/fig6_reporting.py`

训练、图4与图5代码不变。新版主入口新增 `--replot-from`，直接使用已生成的完整图6导出目录，不需要 EEG 缓存、checkpoint 或 GPU。不要仅复制 CSV；需要该目录中的 `node_response_cache` 和审计文件。

```bash
cd ~/autodl-tmp/update-2.0/A-update
set -o pipefail
FIG6_OUT="results/fig6_stew_reporting_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FIG6_OUT"
/root/miniconda3/bin/python -u analysis/fig6_node_response_topomaps.py \
  --replot-from results/fig6_nodes_stew_rebuilt \
  --output-dir "$FIG6_OUT" \
  --color-mode group-detail \
  --color-scope dataset \
  --bootstrap-replicates 5000 \
  --seed 42 \
  --backend auto \
  2>&1 | tee "$FIG6_OUT/run.log"
```

不传 `--replot-from` 时，原有真实 checkpoint 提取流程继续可用，并自动采用新版报告。重绘时分析源目录中的全部数据集；`--datasets` 不用于筛选导出文件。

## 版式与颜色

取消图内长总标题，完整论文标题放 caption。A 为 PRE/POST 组均值与符号一致性；B 为按原 BAcc 规则选定的四人及 Avg；C 为每人相对其他被试的平均相关性 PRE/POST 连线；D 为一致性变化的条件性重采样区间与逐一删除被试范围。

默认 `group-detail`：组 PRE/POST 共享其组均值的对称范围；代表个体与右侧 Avg 使用所有被试 POST 极值定义的独立对称范围，不按所选个体优化色条。两种色条清晰分别标注，同一个 Avg 在两个面板的颜色可能不同。没有单图归一化或裁剪电极极值。`--color-mode shared-original` 恢复全被试 PRE/POST 的原统一色限。

## 统计定义和限制

在共同有效被试对上计算 Pearson r。每次以被试为单位有放回抽样，PRE/POST 使用相同抽样计数；被试对权重为两名被试的抽样次数之积。同一被试与自身的配对排除，避免人工引入 r=1。报告全局 POST−PRE 统计量的2.5%和97.5%分位数。不独立抽样1128对相关系数。

这是以现有训练模型为条件的探索性 subject-weight bootstrap。没有重新训练模型，不能计入 LOSO 重叠训练依赖和训练不确定性；区间不跨零也不作为确认性显著性证据。不生成 p 值、星号或“显著改善”的自动结论。逐一删除被试范围仅反映敏感性，不是置信区间。无效常量图使用两相位共同有效配对，导出有效数。

## 输出

- `Fig6_node_response_topomaps.pdf/png`
- `fig6_color_limits.csv`
- `fig6_exploratory_statistics.csv`
- `fig6_subject_consistency_changes.csv`
- `stew_conditional_bootstrap.csv`（每次重采样的变化值）
- `fig6_summary.md`、`fig6_reporting_notes.md`
- `fig6_replot_provenance.json`（离线重绘输入哈希与参数）

原始 EEG 缓存与 checkpoint 不会被修改；原导出目录不会被覆盖。
