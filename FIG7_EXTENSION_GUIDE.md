# 图7追加计时与结果保留

入口：`analysis/fig7_extended_efficiency.py`。原六模型目录 `results/fig7_fixed_v2` 全程只读，新输出 `results/fig7_extended`。不使用原入口的 `--stage all` 对旧目录重跑。

## 模型与路径

原六模型的路径直接读取旧 `fig7_runs.json`，并检查当前 checkpoint 哈希与历史计时记录一致。目录内旧代码哈希、原始计时数组、504个折的记录均保持原样。新目录的 `fig7_preserved_provenance.json` 记录原结果路径、哈希和历史签名，不将旧记录伪装为新代码测量。

新增路径固定在 `analysis/fig7_additional_run_config.json`：

|数据集|方法|目录|
|---|---|---|
|STEW|BiLSTM|outputs/stew_loso_bilstm|
|EEGMAT|BiLSTM|outputs/eegmat_loso_bilstm|
|STEW|TAHAG|outputs/stew_loso_tahag|
|EEGMAT|TAHAG|outputs/eegmat_loso_tahag|
|STEW|LSCCN|outputs/stew_loso_lsccn|

这些路径与当前上传总表对应。服务器 audit 阶段检查 master/fold summary 一致性和文件存在性，打印并保存实际选择的17条路径。all 阶段进一步严格加载并复算目标 BAcc。本地不能证明服务器磁盘当前状态。

## 接口与公平性

- BiLSTM恢复 recurrent_hidden/layers/dropout，使用双向LSTM原始 EEG 输入。
- TAHAG恢复 dropout/adaptive/attention，使用训练入口相同的 bandpower 特征与无目标批次的评估 forward。
- LSCCN使用训练入口相同 PLV+bandpower 融合特征；二分类恢复各折 summary.csv 的 decision_threshold，按 score1-score0 >= threshold 决策。阈值缺失直接停止，不用测试标签重选。
- 所有新 checkpoint 严格加载；原生目标 BAcc 必须与原 summary 一致；FP32前后所有目标预测相同，并检查batch1预测、参数和缓冲区不变。
- BF-GCN、TAHAG、LSCCN前处理不计入 forward 延迟；LSCCN仍执行完整原 forward，包括重建分支。该指标不是端到端耗时。
- 新旧测量要求同硬件UUID、软件环境、线程和计时预算、相同数据缓存哈希。跨时间GPU负载差异仍需实验者控制，无法从日志证明完全一致。
- 新增216折（STEW 3×48；EEGMAT 2×36）。不计时原来的504折。中断后同命令重跑可复用新模型已完成且签名匹配的折。

## 绘图

沿用此前讨论的正文方案：(a)(b)最终目标BAcc均值±被试间SD；(c)(d)目标BAcc对推理延迟。统一方法顺序、颜色、上下布局、对数延迟坐标；散点标签有引线与避让。不会将不同方法强行连成连续学习曲线，也不插值补造训练历史。旧学习曲线图仍保存在原目录。

STEW显示9种方法、EEGMAT显示8种。EEGMAT LSCCN明确缺失，SVM尚未计时。PDF和600dpi PNG输出到新目录。绘图测试数据仅在pytest临时目录用于排版测试，不能用于论文。

## 服务器运行

```bash
python -u analysis/fig7_extended_efficiency.py --stage audit \
  --existing-results results/fig7_fixed_v2 \
  --run-config analysis/fig7_additional_run_config.json \
  --output-dir results/fig7_extended

python -u analysis/fig7_extended_efficiency.py --stage all \
  --existing-results results/fig7_fixed_v2 \
  --run-config analysis/fig7_additional_run_config.json \
  --cache-root outputs/cache_fig6_rebuilt --device cuda \
  --threads 1 --warmup 100 --repeats 1000 --eval-batch-size 16 --seed 42 \
  --output-dir results/fig7_extended
```

只调整排版时用 `--stage plot --output-dir results/fig7_extended`，不重新计时。新增接口默认仅由扩展入口选用，原六模型主入口不扩充方法名单。
