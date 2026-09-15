# 单被试域自适应、图4/图5及TSMNet原代码核对

## 已实现的协议

训练和两张图新增/使用 `--protocol single_session`。这表示**一个被试内部按每个记录的时间顺序划分**，不是 `--protocol loso --subject 21`。默认每记录约70%训练、10%验证、20%测试。

同一被试三个时间块使用独立统计域。例如 subject21/session1：训练210011、验证210012、测试210013。编号由数据身份和时间块产生，不按类别产生。源正常化器只拟合训练窗口；验证块统计为验证单独重估；最终目标统计只使用无标签测试窗口。目标标签只进入评分和图中事后分组。这是使用整段目标窗口的 transductive adaptation，不是逐窗口在线因果测试。

旧单被试 checkpoint 共用 subject/session 域编号，不能复用为本次独立时间块域实验。新增 `domain_policy=time_blocks_v1` 审计；图4/图5拒绝无该标记的单被试结果。LOSO 保留原来的被试域编号。

图5保持三种对照与第四模型接口，`--fourth-model tsmnet` 可替换第四格。图4增加 `--model tsmnet`，单独计算同一 TSMNet checkpoint 的 PRE/POST 配对机制。图4与图5仍可用 `--protocol loso`，默认就是loso。

Mean-CE及全局SPDBN对照仍保持自己的定义，不人为添加域自适应。EuDSBN、完整AGMNet和TSMNet+SPDDSBN默认开启目标域统计适应；不要传 `--no-target-adapt`。把所有对照都改成域自适应会改变原实验问题。

## 推荐服务器命令

同步本次修改的 `run_experiment.py`、`src/cl_tsmnet/{splits,training}.py`、两张图的analysis脚本，以及 `scripts/run_single_subject_figures.sh`。

先跑被试21，新输出目录保存5个模型；之后分别画AGMNet/TSMNet图4和两版图5：

```bash
cd ~/autodl-tmp/update-2.0/A-update
SUBJECT=21 \
RUN_ROOT=outputs/stew_single_s21_v1 \
RESULT_ROOT=results/stew_single_s21_v1 \
bash scripts/run_single_subject_figures.sh
```

脚本默认30轮、patience8、batch16，明确关闭数据增强以减少适配比较因素。TSMNet默认使用作者 **SPDDSBN constant** 配置。不要覆盖已有LOSO目录。脚本发现已有master summary会停止，避免覆盖checkpoint。

完成单被试检查后，逐个被试做单被试实验，并汇总48个被试：

```bash
SUBJECT=all \
RUN_ROOT=outputs/stew_single_all_v1 \
RESULT_ROOT=results/stew_single_all_v1 \
bash scripts/run_single_subject_figures.sh
```

这里48个评估单元都是各自被试内部的时间块划分，不能写成48折LOSO结果。

仅训练TSMNet：

```bash
/root/miniconda3/bin/python -u run_experiment.py \
  --dataset stew --protocol single_session --subject 21 \
  --model tsmnet --bnorm spddsbn --tsmnet-bn-schedule constant \
  --data-root data --cache-root outputs/cache \
  --output outputs/tsmnet_single_s21_v1 \
  --master-summary outputs/tsmnet_single_s21_v1/master_summary.csv \
  --epochs 30 --patience 8 --batch-size 16 --refit-batch-size 16 \
  --single-val-size 0.125 --test-size 0.2 --seed 42 --no-augment
```

作者主配置SPDDSMBN使用momentum调度，可将上述参数改为 `--tsmnet-bn-schedule momentum --epochs 50`，**换新输出目录**。当前单源域中调度器的bs与bs0均取训练batch-size；不声称复现作者LOSO每批5域的采样设置。momentum要求epochs>11，因为作者调度长度是epochs-10。

只从已有新协议checkpoint重新画图4，不重新训练：

```bash
/root/miniconda3/bin/python -u analysis/fig_spddsbn_paired_mechanism_analysis.py \
  --protocol single_session --model tsmnet \
  --subjects 21 --expected-folds 1 --representative-subject 21 \
  --output-root outputs/tsmnet_single_s21_v1 \
  --master-summary outputs/tsmnet_single_s21_v1/master_summary.csv \
  --output-dir results/tsmnet_single_s21_fig4 --device cpu --batch-size 16
```

已有导出和指标时用 `--stage analyze`，仍须指定匹配的 `--protocol single_session --model tsmnet`。完整AGMNet对应 `--model ms_tgc_spddsbn`。图4不支持无SPDDSBN的裸TSMNet，因为该模型没有待比较的层。

## 与作者原仓库的差异

核对仓库：https://github.com/rkobler/TSMNet ，固定提交 `90293b9d2982fa06a3030340d24d287155fd5a89`。逐文件差异保存为 `TSMNET_UPSTREAM_DIFF.patch`（不含pyc）。

| 项目 | 核对结果 |
|---|---|
| spdnets/models/tsmnet.py | 与作者版本相同：时空卷积、协方差池化、BiMap、ReEig、SPDDSBN、LogEig、线性分类器 |
| modules.py、manifolds.py | 与作者版本相同 |
| batchnorm.py | 本地有skorch可选导入，以及域统计buffer形状/别名的兼容修复，不能称为逐字原版 |
| functionals.py | 本地增加Karcher均值收敛诊断和可选容差；训练默认容差仍用原EPS |
| 优化器 | 采用原RiemannianAdam及W/mean不衰减参数组；此次取消缺geoopt时降级普通Adam |
| BN调度 | 原本训练入口未调用作者调度器；此次加入原ConstantMomentumBatchNormScheduler及MomentumBatchNormScheduler，直接调用其实现 |
| 数据/训练入口 | 当前STEW窗口、预处理、时间划分、增强、验证和提前停止都是项目适配；不等于作者Hydra/skorch实验 |

这些兼容修复关系到现有AGMNet和已保存checkpoint，不整体回滚公共spdnets。**严格原代码实验通过下面独立入口运行**，不让修改后的共享代码伪装成原版。

## 执行严格作者原代码

作者原实验支持BNCI2014001、BNCI2015001、Lee2019、Stieger2021、Hinss2021；没有STEW适配器。原代码入口不能直接传 `dataset=stew`。这与上述“原TSMNet结构用于STEW适配实验”是不同的实验。

新增 `run_tsmnet_original.py` 会下载并核对固定提交，然后用当前Python执行作者原始 `experiments/main.py`，代码不经本项目训练器。先准备源码，再在**独立环境**安装作者固定依赖。环境文件有 `pip -e .`，所以必须在该仓库目录创建环境：

```bash
cd ~/autodl-tmp/update-2.0/A-update
/root/miniconda3/bin/python run_tsmnet_original.py --prepare
cd external/TSMNet-paper
conda env create -n tsmnet-paper-original -f environment.yaml
cd ../..
conda run -n tsmnet-paper-original python run_tsmnet_original.py -- \
  dataset=bnci2014001 evaluation=inter-session+uda
```

原环境固定Python3.8、PyTorch1.9、旧版numpy/scipy和geoopt提交；不在当前Python3.12服务器环境中覆盖安装。作者入口会下载其数据并运行原实验，不产生当前图4/5的STEW格式checkpoint。原始数据与本项目STEW结果不能混用。

## 验证范围

本地已验证：单被试划分与绘图域编号完全一致；TSMNet/AGMNet真实前后向训练、验证及目标统计适应、checkpoint保存/严格重载、图4导出及图5特征提取；源统计重估保持目标预测。另以默认40→20维TSMNet完成12轮momentum调度训练并检查损失有限。

上述测试使用明确的数值测试数据，不是STEW科学结果。已核验原仓库版本与源文件差异，但未在本地安装其旧环境或执行完整原论文数据实验；不能保证当前服务器所有折都不会遇到eigh数值故障。若出现异常请保留完整日志，不能将普通Adam、裁剪矩阵或替换模型作为未经说明的修复。
