# LEOD 项目生产手册

## 1. 文档目的

本文档用于指导 `EventDet-EFM / LEOD` 项目在本地 Windows 单机环境中的稳定运行，覆盖以下内容：

- 环境基线
- 数据集落地规范
- 训练、验证、可视化标准流程
- 日志与产物目录约定
- 常见故障定位与处置

本文档不是论文说明，也不是算法综述，而是面向“把项目稳定跑起来并持续产出结果”的操作手册。

## 2. 适用范围

当前手册按以下已验证环境编写：

- 操作系统：Windows
- Python 环境：Miniforge / Conda
- Conda 环境名：`myenv`
- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`
- 显存：`6 GB`
- PyTorch：`2.7.0+cu128`
- CUDA：`12.8`
- 数据集：`Gen1` 预处理版

如果后续更换 Linux、多卡、A40 / 4090 等更大显存设备，本手册中的“资源约束”和“推荐命令”需要重新校准。

## 3. 项目现状结论

当前仓库已经完成以下关键打通工作：

- `Gen1` 数据集可正常读取
- `myenv` 环境可正常使用 CUDA
- `train.py` 可以在本机完成训练启动
- 1-step 冒烟训练已通过
- 训练链路和验证链路均已在本机验证通过

基于当前环境，最稳定的训练入口不是论文原始默认配置，而是针对本机资源收敛后的轻量配置：

```powershell
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss +experiment/gen1=tiny_win3060.yaml
```

## 4. 目录规范

### 4.1 输入目录

- 数据集根目录：`datasets/gen1/`
- 训练数据：`datasets/gen1/train/`
- 验证/测试数据：`datasets/gen1/test/`

### 4.2 输出目录

- 训练 checkpoint：`checkpoint/`
- TensorBoard 日志：`tb_logs/`
- 验证日志：`validation_logs/`
- 可视化视频：`vis/`

### 4.3 关键配置文件

- 安装说明：[docs/install.md](./install.md)
- 论文实验命令：[docs/benchmark.md](./benchmark.md)
- 本机稳定训练配置：[config/experiment/gen1/tiny_win3060.yaml](../config/experiment/gen1/tiny_win3060.yaml)
- 训练入口：[train.py](../train.py)
- 验证入口：[val.py](../val.py)
- 伪标签生成入口：[predict.py](../predict.py)
- 视频可视化入口：[vis_pred.py](../vis_pred.py)

## 5. 环境基线

### 5.1 启动环境

新开 PowerShell 后应自动激活 `myenv`。若未自动激活，手动执行：

```powershell
conda activate myenv
```

### 5.2 基础核验

每次重要训练前建议执行：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

预期结果应满足：

- `torch.cuda.is_available()` 为 `True`
- 能识别到 `NVIDIA GeForce RTX 3060 Laptop GPU`

### 5.3 数据集核验

```powershell
python -c "from pathlib import Path; root=Path('datasets/gen1'); print(root.exists(), (root/'train').exists(), (root/'test').exists())"
```

预期输出应为：

- `True True True`

## 6. 运行模式说明

### 6.1 为什么不用论文默认训练配置

论文默认配置更偏向 Linux / Slurm / 更大资源环境。在当前 Windows + 6GB 显存设备上，默认配置存在以下风险：

- `small.yaml` 显存压力较大，容易 OOM
- `mixed` 采样依赖多进程 DataLoader，在 Windows 下不稳定
- 高维日志和部分可视化回调默认面向 WandB，不适合作为本机最小稳定路径

因此当前本机训练统一采用：

- `tiny` 模型尺寸
- `batch_size.train=1`
- `batch_size.eval=1`
- `hardware.num_workers.train=0`
- `hardware.num_workers.eval=0`
- `dataset.train.sampling=random`

## 7. 标准操作流程

### 7.1 冒烟测试

在正式训练前，先验证训练与验证链路能否启动：

```powershell
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss +experiment/gen1=tiny_win3060.yaml training.max_steps=1 validation.val_check_interval=1 validation.limit_val_batches=1
```

通过标准：

- 能进入 `Sanity Checking`
- 能完成至少 1 个训练 step
- 能执行 1 次验证
- 最终出现 `Trainer.fit stopped: max_steps=1 reached.`

### 7.2 正式训练

本机正式训练标准命令：

```powershell
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss +experiment/gen1=tiny_win3060.yaml
```

### 7.3 恢复训练

当前训练脚本会自动从同名实验目录下的最近 checkpoint 续跑。

当目录中已存在历史权重时，训练启动日志可能出现：

```text
INFO: automatically detect checkpoint ...
```

说明程序正在自动恢复训练状态。

如果想“重新开始新训练”，先清理对应实验目录，例如：

- `checkpoint/rnndet_tiny-gen1-bs1_iter0k/`

不要在不确认的情况下删除整个 `checkpoint/`。

### 7.4 验证已有 checkpoint

验证命令模板：

```powershell
python val.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ checkpoint="你的ckpt路径" use_test_set=1 hardware.gpus=0 hardware.num_workers.eval=0 +experiment/gen1=tiny_win3060.yaml batch_size.eval=1 model.postprocess.confidence_threshold=0.001 reverse=False tta.enable=False
```

说明：

- `val.py` 当前按单卡 GPU 写法运行
- 本机建议 `batch_size.eval=1`
- 若验证 OOM，先不要提高 batch

### 7.5 可视化预测结果

```powershell
python vis_pred.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ checkpoint="你的ckpt路径" +experiment/gen1=tiny_win3060.yaml model.postprocess.confidence_threshold=0.1 num_video=5 reverse=False
```

输出视频默认保存在 `vis/` 下。

## 8. 资源预估

### 8.1 显存

在 `RTX 3060 Laptop 6GB` 上：

- `tiny_win3060.yaml` 可稳定训练
- `small.yaml` 不建议作为默认首选

### 8.2 训练耗时

实测训练速度大约在：

- `1.2 it/s` 左右

而当前配置默认：

- `training.max_steps=400000`

粗略耗时：

- `400000 / 1.24 ≈ 89.6 小时`
- 约等于 `3.7 天`

这只是估算值，实际会受：

- 温度墙
- 后台占用
- 是否触发验证
- 磁盘 IO

影响。

### 8.3 Epoch 解释

本项目更应关注 `max_steps`，而不是自然语言理解里的“跑完一个 epoch”。

原因是：

- 单个 epoch 的 step 数很大
- 本机单步速度较慢
- 训练停止条件主要由 `max_steps` 控制

## 9. 日志判读

### 9.1 loss 下降意味着什么

`loss` 持续下降通常表示：

- 模型正在学习
- 参数更新有效
- 训练没有卡死

但 `loss` 下降不等于最终检测性能一定提升。

真正需要结合看的还有：

- 验证指标
- 可视化结果
- 是否出现过拟合

### 9.2 关键观察项

训练过程中建议重点关注：

- 是否持续报错
- 是否出现 OOM
- `loss` 是否总体下降
- checkpoint 是否定期生成
- TensorBoard 曲线是否异常震荡或恒定不变

## 10. 常见故障与处理

### 10.1 显存不足

典型表现：

- `CUDA out of memory`

处理顺序：

1. 确认使用 `tiny_win3060.yaml`
2. 关闭其他占用 GPU 的程序
3. 不要上调 `batch_size`
4. 暂时不要切换到 `small.yaml`

### 10.2 Windows 下 mixed 采样报错

典型表现：

- `Cannot use mixed mode with batch size smaller than 2`
- `Cannot use mixed mode with num workers smaller than 2`
- `Can't pickle local object ...`

原因：

- `mixed` 采样依赖多进程 / DataPipe 组合，在当前 Windows 环境下不稳定

处置：

- 统一使用 `dataset.train.sampling=random`
- 统一使用 `hardware.num_workers.train=0`

### 10.3 自动续跑旧 checkpoint

典型表现：

- 明明想重训，却自动恢复到了旧 step

原因：

- [`train.py`](../train.py) 会自动扫描实验目录里的最近 checkpoint

处置：

- 清理对应实验目录
- 或切换新的实验名 / 配置组合

### 10.4 验证或可视化阶段崩溃

优先检查：

- `checkpoint` 路径是否正确
- `dataset.path` 是否正确
- `hardware.gpus=0` 是否可用
- `batch_size.eval` 是否过大

### 10.5 训练非常慢

当前本机慢是正常现象，不应优先怀疑代码错误。影响速度的核心因素包括：

- 6GB 显存导致 batch 只能为 1
- Windows 下不使用多 worker
- 数据集规模较大

## 11. 交付物标准

一次完整训练任务，至少应保留以下内容：

- 可复现实验命令
- 对应配置文件名
- 最终 checkpoint 路径
- TensorBoard 日志目录
- 验证结果记录
- 关键截图或可视化样例

建议在论文或阶段汇报中统一记录以下字段：

- 数据集版本
- 模型配置
- 训练步数
- 硬件环境
- 最佳 checkpoint
- 验证指标

## 12. 推荐最短命令清单

### 12.1 正式训练

```powershell
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss +experiment/gen1=tiny_win3060.yaml
```

### 12.2 训练冒烟验证

```powershell
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss +experiment/gen1=tiny_win3060.yaml training.max_steps=1 validation.val_check_interval=1 validation.limit_val_batches=1
```

### 12.3 验证 checkpoint

```powershell
python val.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ checkpoint="你的ckpt路径" use_test_set=1 hardware.gpus=0 hardware.num_workers.eval=0 +experiment/gen1=tiny_win3060.yaml batch_size.eval=1 model.postprocess.confidence_threshold=0.001 reverse=False tta.enable=False
```

### 12.4 可视化预测

```powershell
python vis_pred.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ checkpoint="你的ckpt路径" +experiment/gen1=tiny_win3060.yaml model.postprocess.confidence_threshold=0.1 num_video=5 reverse=False
```

## 13. 后续文档建议

如果后续要把这份材料并入毕设正文，建议作为以下两部分的底稿：

- “实验环境与实现细节”
- “工程实现与复现实验流程”

论文正文写法应偏学术表达，而本手册保留工程和运维视角，两者不要混写。
