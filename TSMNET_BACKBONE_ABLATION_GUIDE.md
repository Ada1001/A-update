# 独立 TSMNet 骨干消融与图4/图5

新增入口：`analysis/tsmnet_backbone_ablation.py`；新增模型：`src/cl_tsmnet/tsmnet_backbone_ablation.py`。没有修改原训练入口、AGMNet模型、图4/图5脚本或原实验文件。新入口只读复用原数据划分和分析函数，所有新结果写入指定新目录。

## 模型定义

所有分支共享同一结构、同一随机种子初始化的原TSMNet CNN：4个时域滤波器（核25）、40个空间滤波器。空间卷积跨所有电极，输出 `[batch,40,time]`。CNN中不额外添加激活、图卷积、通道注意力或128维隐藏分类头。不同分支独立训练，不共享训练后的权重。

| 名称 | CNN后处理 | 线性分类器输入 | 目标适应 |
|---|---|---|---|
| mean-ce | 时间均值 | 40维 | 无 |
| mean-eudsbn | 时间均值→EuDSBN | 40维 | 目标窗口统计重估 |
| augspd-spdbn | 均值/协方差构造41×41 AugSPD→BiMap/ReEig→20×20→全局SPDBN→LogEig | 210维 | 不用目标统计 |
| tsmnet | 原TSMNet协方差路径→BiMap/ReEig→SPDDSBN→LogEig | 210维 | 目标窗口统计重估 |
| augspd-spddsbn | 与augspd-spdbn同一AugSPD路径，换成SPDDSBN | 210维 | 目标窗口统计重估 |

AugSPD采用项目已验证的构造：`[[C + μμᵀ, μ], [μᵀ, 1]]`，协方差shrinkage=0.1、epsilon=1e-5。原TSMNet继续用自己的CovariancePool，不替换成AugSPD。

EuDSBN是本次新分支，采用各域独立统计、**跨域共享可学习仿射参数**，避免目标域使用从未训练过的独立gamma/beta。这个实现选择与原AGMNet的EuDSBN不是同一消融，论文需要说明。

`tsmnet`指原网络的 **TSMNet+SPDDSBN** 版本，采用constant动量eta=eta_test=0.1；不等同于无BN裸TSMNet，也不等同于作者主配置的SPDDSMBN。使用项目现有TSMNet代码及其buffer兼容修复；不是声称完整复现作者数据/训练协议。

## 消融解释

- mean-ce vs mean-eudsbn：同一均值表示，检查EuDSBN作用。
- augspd-spdbn vs augspd-spddsbn：同一AugSPD表示，检查域特异归一化作用。
- augspd-spddsbn vs tsmnet：比较本项目AugSPD路径与原TSMNet协方差路径；同时存在协方差正则化选择差异，不能仅归因于均值增强。
- augspd-spdbn vs tsmnet同时改变表示与归一化，不能作为严格的单因素BN消融。

默认图5四格依次为前三个分支和原TSMNet。可另外生成第四格为augspd-spddsbn的版本，以支撑归一化消融。

原TSMNet CNN是线性卷积，EEG常接近零均值，因此mean分支可能较弱。不会为了让均值分支结果更好而插入ReLU或挑种子。二维散点只用于描述；图5右侧指标使用全部高维样本。不同网络学习到的SPD空间不相同，图4的绝对AIRM数值不能直接充当跨模型优劣的统一尺度。

## 训练和审计

- 支持`single_session`和`loso`。单被试默认210/30/60窗口时序划分，三块统计域分开；LOSO为按被试隔离的训练/验证/测试。
- 训练器CPU运行，CNN float32，SPD/线性分类头float64；默认无数据增强、无异常窗口过滤。
- 全分支使用相同的无标签按域组批顺序。每批一个域；尾部单窗口并入前批，因此偶尔为batch-size+1。RiemannianAdam不静默降级普通Adam。
- 验证域只使用无标签验证窗口重估统计，验证标签仅用于early stopping；验证统计恢复后继续训练。全局SPDBN仅用源训练窗口重估。
- 选出最优验证checkpoint后，适应目标统计，保存checkpoint。源训练统计诊断在内存副本状态上进行，不覆盖checkpoint。校验只有源统计buffer变化，目标logits容差内一致、预测完全一致。
- 目标标签不输入refit；属于整段无标签目标窗口可用的transductive评估。
- PRE/POST、标签、样本ID、域、特征、logits同时导出；检查SPD对称性、正定性及文件哈希。

## 服务器：单被试21先完整跑通

同步新增的模型文件与analysis入口（保留项目已有依赖），在项目根目录运行：

```bash
cd ~/autodl-tmp/update-2.0/A-update
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/root/miniconda3/bin/python -u analysis/tsmnet_backbone_ablation.py \
  --stage all --protocol single_session --datasets stew --dataset-labels STEW \
  --subjects 21 --representative-subject 21 \
  --epochs 30 --patience 8 --batch-size 16 --threads 1 \
  --seed 42 --data-root data --cache-root outputs/cache \
  --fourth-model tsmnet --reducer tsne \
  --output-dir outputs/tsmnet_backbone_single_s21_v1
```

这会训练五个分支，画前三个消融与原TSMNet的图5，以及原TSMNet和AugSPD-SPDDSBN各自的图4。不修改 `outputs/stew_single_s21_v1` 等旧实验。

追加严格BN对照图5，不重训：

```bash
/root/miniconda3/bin/python -u analysis/tsmnet_backbone_ablation.py \
  --stage plot --figures fig5 --protocol single_session \
  --representative-subject 21 --fourth-model augspd-spddsbn --reducer tsne \
  --output-dir outputs/tsmnet_backbone_single_s21_v1
```

单独运行原TSMNet训练和图4：

```bash
/root/miniconda3/bin/python -u analysis/tsmnet_backbone_ablation.py \
  --stage all --figures fig4 --models tsmnet \
  --protocol single_session --subjects 21 --representative-subject 21 \
  --epochs 30 --batch-size 16 \
  --output-dir outputs/tsmnet_original_single_s21_independent_v1
```

## 所有被试与LOSO

单被试协议覆盖所有被试，将首条命令改为`--subjects all`并使用新目录`outputs/tsmnet_backbone_single_all_v1`。

LOSO全部折建议分阶段：

```bash
/root/miniconda3/bin/python -u analysis/tsmnet_backbone_ablation.py \
  --stage train --protocol loso --subjects all --epochs 30 --batch-size 16 \
  --output-dir outputs/tsmnet_backbone_loso_v1

/root/miniconda3/bin/python -u analysis/tsmnet_backbone_ablation.py \
  --stage plot --protocol loso --representative-subject 21 \
  --fourth-model tsmnet --reducer tsne \
  --output-dir outputs/tsmnet_backbone_loso_v1
```

图4使用全部样本计算精确AIRM邻居，LOSO计算可能很慢，不能把`max_pairs=50000`误解为kNN也只计算这些对。图4每折指标可复用已有分析缓存；重跑`--stage plot`不会训练。训练阶段不覆盖已存在checkpoint，中断训练后不要把其他配置文件拼进目录，另用新目录重跑。

## 已有原TSMNet实验的可视化

已有 `outputs/stew_single_s21_v1` 无需迁移到新训练器，可继续用未修改的旧入口单独输出新图目录：

```bash
/root/miniconda3/bin/python -u analysis/fig_spddsbn_paired_mechanism_analysis.py \
  --protocol single_session --model tsmnet --subjects 21 --expected-folds 1 \
  --representative-subject 21 \
  --output-root outputs/stew_single_s21_v1 \
  --master-summary outputs/stew_single_s21_v1/master_summary.csv \
  --output-dir results/original_tsmnet_s21_fig4_review

/root/miniconda3/bin/python -u analysis/fig5_representation_alignment.py \
  --protocol single_session --datasets stew --dataset-labels STEW \
  --fourth-model tsmnet --target-subjects stew=21 \
  --output-root outputs/stew_single_s21_v1 \
  --master-summary outputs/stew_single_s21_v1/master_summary.csv \
  --source-calibration refit --reducer tsne \
  --output-dir results/original_tsmnet_s21_fig5_review
```

此处旧入口的前三格仍是原AGMNet骨干对照；要看TSMNet骨干的前三个消融，必须用本文的新入口，不能把旧checkpoint改名冒充新骨干。

## 输出与验证范围

新实验目录含：`experiment.json`、`summary.csv`；每模型每被试含`model.pt`、`history.csv`、`training_config.json`、`preprocessing.npz`、`scores.json`、`audit.json`、`samples.csv`、`paired_spd.npz`。均值分支的archive只含特征和logits，没有伪造SPD矩阵。

图5在`fig5_tsmnet/`或`fig5_augspd-spddsbn/`，包含600dpi PNG、矢量PDF、坐标与高维指标CSV。图4在`tsmnet/`和`augspd-spddsbn/`，含配对指标CSV与报告。多个评估单元的图5误差条表示标准差，不是置信区间；单折不画误差条。

本地测试数据仅用于验证五分支训练、梯度有限、同CNN初始化、分域组批、checkpoint严格重载、目标预测不变、原TSMNet图4指标和图5渲染。没有用示意数据冒充真实STEW实验，也没有据测试通过保证服务器所有真实数据都不发生数值故障。
