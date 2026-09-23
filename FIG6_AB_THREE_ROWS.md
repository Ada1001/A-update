# 图6 A/B三行版

上传 `analysis/fig6_node_response_topomaps.py` 和 `analysis/fig6_reporting.py` 后执行：

```bash
cd ~/autodl-tmp/update-2.0/A-update
/root/miniconda3/bin/python -u analysis/fig6_node_response_topomaps.py \
  --replot-from results/fig6_nodes_stew_rebuilt \
  --output-dir results/fig6_stew_ab_three_rows \
  --layout ab-three-rows \
  --color-mode group-detail \
  --color-scope dataset \
  --bootstrap-replicates 5000 --seed 42 --backend auto
```

输入是原始完整导出目录，不是只有重绘图片的v2目录。无需GPU、EEG缓存或重新训练。

- 仅A/B，三行依次High−Low、High、Low。A为组平均PRE/POST；B为原四名代表被试POST及全被试Avg。
- 原符号一致性列取消：高低负荷L2响应本身非负，对它们画符号一致性没有对应分析价值。
- 三行统一使用红蓝配色 RdBu_r；全部色条竖放。差异图红蓝表示正负对比；高低负荷图红蓝表示响应较高或较低，不表示正负值。
- 同一面板High/Low共用色限，A中的PRE/POST共用色限，不做单图归一化。
- 默认B范围由展示的四人和Avg定义；本次差异色限±15.5172，原图为±45.2662。该显示范围改变写入CSV和报告，不代表统计显著性改变。展示的电极数值没有截断。
- A/B色限不同，不能直接跨面板比较颜色深浅。High/Low色条下限不强制为零，仍显示真实L2均值。
- 探索性统计仍导出CSV，但C/D不出现在图中。原统计图可通过 `--layout diagnostic` 生成。
- 若要恢复全被试极值范围，加 `--color-mode shared-original`。

输出PNG、PDF、`fig6_color_limits.csv`、统计CSV及报告。原始导出目录不覆盖。

## 额外全被试图

同一运行命令还自动保存 `Fig6_all_subjects_stew_post_contrast.pdf/png`。
按被试ID排序，每行最多8人，仅展示POST的High−Low，不增加Avg或高低负荷单独行。
全部被试默认固定使用−10到10的红蓝色限，可用 `--all-subjects-limit 10` 显式指定。
超范围数值仅在颜色上饱和，原始数值与统计不修改；色条两端箭头表示范围延伸。超范围电极值数量和原始极值均导出CSV。此参数只影响新增全被试图，不改变主A/B图。
竖向色条全图共用，范围另存 `fig6_all_subjects_color_limits.csv`。如果输入只是被试子集，图中只包含该子集。
