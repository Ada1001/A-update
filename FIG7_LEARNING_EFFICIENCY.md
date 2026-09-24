# Fig.7：Learning and Computational Efficiency

入口：`analysis/fig7_learning_efficiency.py`；模型重建/FP32推理副本：`analysis/fig7_models.py`。
`src/cl_tsmnet/training.py` 仅增加逐epoch日志和来源记录，不改优化器、early stopping准则或训练精度。

## 已确认的输入情况

本地 `outputs/master_summary.csv` 包含 STEW/EEGMAT 的 EEGNet、EEG-Conformer、BF-GCN、TSMNet、AGMNet LOSO汇总；未发现canonical MDTN-GMDA记录。AGMNet有同目录多条历史记录及独立消融目录，不能直接混在一起。
本地没有这些正式实验的全部fold文件；需在服务器扫描真实目录。不要把本地缺文件误认为服务器也缺。不要上传本地生成的补跑shell脚本；应在服务器重新执行audit生成。

旧训练已有 `history.csv`（epoch/train_loss/val_loss/val_bacc）。旧 `val_bacc` 是旧运行代码中的验证指标，不是总表的最终val_bacc。没有每epoch日志时，总表无法恢复曲线。旧日志缺训练秒数时保留NaN，不伪造时间，也不因只缺秒数而重训。

## 服务器操作

同步三个Python文件后，先审计。下例假设根目录是 `outputs`；若实际为 `output`，只改 `RUNS=output`。MASTER须指向真实服务器总表。

```bash
cd ~/autodl-tmp/update-2.0/A-update
RUNS=outputs
MASTER="$RUNS/master_summary.csv"
/root/miniconda3/bin/python -u analysis/fig7_learning_efficiency.py \
  --stage audit --master-summary "$MASTER" --output-root "$RUNS" \
  --cache-root outputs/cache_fig6_rebuilt \
  --repair-root outputs/fig7_retrained \
  --output-dir results/fig7_audit
```

检查 `fig7_input_audit.json`、`fig7_runs.json`、`fig7_repair_plan.json`。审计阶段不启动训练。若某方法有多个不同目录，先写run-config明确选择，例如：

```json
{"stew":{"ms_tgc_spddsbn":"outputs/stew_loso_ms_tgc_spddsbn"},
 "eegmat":{"ms_tgc_spddsbn":"outputs/eegmat_loso_ms_tgc_spddsbn"}}
```

必要时在audit命令加 `--run-config 文件.json` 再审计。不得根据最高target BAcc挑选实验；按预先确定的配置选择。

缺checkpoint或epoch日志时，生成的shell会补跑该方法全部fold到独立目录，拒绝覆盖已有目录：

```bash
bash results/fig7_audit/fig7_retrain_missing.sh
```

脚本复用该方法总表中已记录的训练参数；缺失MDTN使用同数据集EEG-Conformer的公共训练协议及当前MDTN默认结构。未记录的历史实现/超参数不能保证完全重建，计划JSON明确记录来源，应先检查。遗留架构checkpoint加载不兼容时必须使用对应原代码或补跑，不能仅重命名权重。已有日志不会为了补齐elapsed秒数而重训。

补齐后执行（在同一台机器独占/空闲GPU上计时，避免同时训练）：

```bash
RUNS=outputs
MASTER_LIST="$RUNS/master_summary.csv"
if [ -f outputs/fig7_retrained/master_summary.csv ]; then
  MASTER_LIST="$MASTER_LIST,outputs/fig7_retrained/master_summary.csv"
fi
set -o pipefail
mkdir -p results/fig7
/root/miniconda3/bin/python -u analysis/fig7_learning_efficiency.py \
  --stage all --master-summary "$MASTER_LIST" --output-root "$RUNS" \
  --run-config results/fig7_audit/fig7_repaired_run_config.json \
  --data-root data --cache-root outputs/cache_fig6_rebuilt \
  --device cuda --threads 1 --warmup 100 --repeats 1000 \
  --eval-batch-size 16 --bootstrap-replicates 5000 --seed 42 \
  --trust-legacy-source-validation \
  --output-dir results/fig7 \
  2>&1 | tee results/fig7/run.log
```

`--trust-legacy-source-validation`仅适用于你已确认旧训练代码用源域验证集选择checkpoint的情况；它是显式的旧日志来源声明，不是自动审计通过。未知旧来源应补跑。新版日志有 `source_validation_audit.json`，会检查验证/目标被试隔离及选择指标，错误时停止。审计不能仅凭一个CSV证明历史代码从未使用目标标签。

可分阶段使用 `--stage convergence`、`benchmark`、`plot`。计时结果按fold存储并校验checkpoint/数据/代码/环境/参数哈希；相同条件中断后重跑可复用完成的fold。变化时使用新输出目录。全部fold都测量，不按性能挑选计时checkpoint。

## 科学与工程边界

- 收敛曲线4方法，trade-off 6方法；完整图必须两个数据集均齐全，不画虚构占位点。
- 每epoch只用实际记录，>=80%fold才绘制；不carry forward。bootstrap CI是固定已训练fold的描述性区间，LOSO重叠训练不支持独立重复推断。
- early stopping和checkpoint选择保持当前代码的**源验证loss最小**准则；图展示验证BAcc，不改成按BAcc选择。
- 每折FP32模型用同一个固定真实目标窗口计时。batch1，eval/no_grad，预热100次、正式1000次；CUDA每次前后同步，计时包括CPU发起forward与GPU完成，不是吞吐量。
- BF-GCN的forward输入是从该1秒窗预计算的bandpower和PLV；预处理不计入forward。图下注明，不能声称其端到端原始EEG处理同样快。
- 原TSMNet/AGMNet是FP32前端+CPU/FP64 SPD。为了满足统一FP32，独立推理副本显式将SPD数学移到所选设备和FP32。**这是FP32推理移植的测量，不是原生混合精度耗时**，不改训练代码或保存的权重。
- FP32验证要求全部目标窗口预测与原生一致，原生BAcc与summary一致；任一失败停止。浮点tensor运算精度/设备也被审计。不能为了画点忽略SPD数值不稳定。
- 本地CPU接口测试通过不代表RTX4090D实测；硬件、PyTorch、CUDA、cuDNN版本均保存。非4090D时明确warning，不能标成4090D结果。
- 参数量只计requires_grad=True，图中面积有明确缩放上限。只画真实点，不预设Pareto关系或更优结论。
- AGMNet图传播实际为dense einsum，不能因top-k掩码就宣称稀疏复杂度；详细说明自动写入分析Markdown。

## 输出

`Fig7_learning_computational_efficiency.pdf/png`（600dpi、PDF矢量曲线和文字）；
`fig7_convergence_epochwise.csv`、`fig7_convergence_summary.csv`、`fig7_convergence_per_fold.csv`、`fig7_epoch_records.csv`；
`fig7_efficiency_summary.csv`、`fig7_efficiency_per_fold.csv`；
`FIG7_EFFICIENCY_ANALYSIS.md`、环境/输入/绘图审计JSON及每fold原始延迟数组。

CSV BAcc以0–1保存，图仅乘100转换百分数。latency_summary从每fold相同次数的实际计时合并计算median/mean/SD/IQR；target BAcc为fold等权均值和样本SD。
