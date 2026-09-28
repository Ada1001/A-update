# 图7主图与补充图

新增入口 `analysis/fig7_learning_reporting.py`，只读取已有训练日志、checkpoint哈希和已完成计时CSV，不训练、不计时、不修改输入结果。

## 输出

- `Fig7_learning_and_efficiency.pdf/png`：上排STEW七种、EEGMAT六种方法的真实源验证BAcc学习曲线；下排STEW九种、EEGMAT八种方法的已有目标BAcc/延迟。
- `FigS7_final_performance.pdf/png`：原扩展版的最终目标BAcc均值与被试间SD、效率散点。
- `fig7_validation_protocol_audit.csv`：逐折验证统计重估、LSCCN最终阈值、来源sidecar、best epoch等。
- `fig7_convergence_epochwise.csv`、`fig7_epoch_records.csv`：真实曲线指标、覆盖率和原始日志汇总。
- `FIG7_TWO_VERSIONS_REVIEW.md`、`fig7_reporting_audit.json`、`fig7_learning_sources.json`：检查结论、输入哈希、历史日志来源。

学习曲线方法：BiLSTM、TAHAG、LSCCN（仅STEW）、EEG-Conformer、MDTN-GMDA、TSMNet、AGMNet。不虚构EEGMAT LSCCN，也不增加SVM的epoch曲线。

## 检查和解释

模型路径沿用原六模型的 `fig7_runs.json` 和扩展的 `fig7_additional_runs.json`。每个学习日志所在目录的checkpoint哈希必须匹配已计时的checkpoint，完整被试集合、summary与学习历史的epoch数及best loss/epoch必须一致。并验证历史文件与epoch_metrics.csv共同存在时的核心字段是否冲突。

LSCCN当前训练代码逐epoch用阈值0验证，最终评估才使用验证集选择的阈值。不会将最终阈值重新应用到缺少原始预测的历史曲线。TAHAG当前验证用eval模式、源验证频带特征、无目标批次；BiLSTM用归一化原始窗口。当前代码不能证明旧实验使用相同版本。

旧日志若无来源sidecar，默认停止；`--trust-legacy-source-validation` 是用户对旧实验来源的明确声明，输出仍标注未独立验证。不能仅因模型名称相同就推断验证协议相同。TSMNet/AGMNet缺失val_stat_refit标志按unknown处理。

只画至少80%折仍有真实日志的epoch，采用实际epoch连接折线；早停后不补齐、不前向填充、不做平滑。bootstrap区间描述当前LOSO折，不能视为独立人群重复实验。E95基于各折自身峰值，不能单独证明公平的学习速度优势。缺少真实训练时间时不生成耗时收敛曲线。

效率图横轴采用原汇总表的 `latency_median_ms`，是合并各折正式测量后的中位数，不是平均数。GPU UUID不同以及TAHAG两折BAcc差异会保留为解释限制；接受容差并没有证明差异来自CSV精度。两个版本复用同一套现有结果，不重新估计延迟或修改BAcc。

## 服务器命令

同步 `analysis/fig7_learning_reporting.py` 和 `analysis/fig7_extended_efficiency.py`，保留已有主脚本和模型适配器。使用包含真实历史训练目录的服务器运行：

```bash
mkdir -p results/fig7_two_versions
set -o pipefail
/root/miniconda3/bin/python -u analysis/fig7_learning_reporting.py \
  --original-results results/fig7_fixed_v2 \
  --extended-results results/fig7_extended_v2 \
  --output-dir results/fig7_two_versions \
  --bootstrap-replicates 5000 --seed 42 \
  --trust-legacy-source-validation \
  2>&1 | tee results/fig7_two_versions/run.log
```

仅检查日志而不生成图时添加 `--stage audit`；不需要GPU参数。复跑此入口不会触发旧计时缓存的代码版本检查。原结果目录及原图片不变。若旧日志无验证BAcc，不用训练/测试准确率替换，也不能从最终权重恢复缺失历史。
