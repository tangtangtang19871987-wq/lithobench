# LithoBench PyTorch -> TensorFlow 迁移清单（进行中）

## 1) 仓库框架梳理
- 入口脚本：
  - `lithobench/train.py`：按 `-m` 选择模型，区分 `ILT` 与 `Litho` 两类任务。
  - `lithobench/test.py`：测试/评估入口，支持 `MetalSet/ViaSet/StdMetal/StdContact`。
- 基类接口：
  - `lithobench/model.py::ModelILT`
  - `lithobench/model.py::ModelLitho`
  - 关键接口：`pretrain/train/run/save/load/evaluate`
- 数据与预处理：
  - `lithobench/dataset.py`：读取 `work/<dataset>` 下 png/glp，返回 PyTorch `NCHW` 张量。
  - 数据增强：随机 crop + flip（训练集）。
- 物理仿真耦合：
  - `pylitho.exact.LithoSim` 在多个模型训练中参与 loss 计算（非纯 DNN）。

## 2) 神经网络家族（重点）
- ILT:
  - `ilt/neuralilt.py`：U-Net 风格（Conv+BN+ReLU, MaxPool, Bilinear Upsample, Skip）
  - `ilt/ganopc.py`：GAN（生成器+判别器）
  - `ilt/damoilt.py`：U-Net + GAN 判别器
  - `ilt/cfnoilt.py`：CFNO（复数频域层 + CNN 分支）
- Litho:
  - `litho/lithogan.py`：GAN（双输出：aerial/resist）
  - `litho/doinn.py`：RFNO + CNN 分支
  - `litho/damolitho.py`：U-Net + GAN 判别器
  - `litho/cfnolitho.py`：CFNO 变体 + 双输出

## 3) PyTorch 与 TensorFlow 常见不一致点（迁移风险清单）
- 张量格式：PyTorch `NCHW` vs TF `NHWC`（必须统一转换）
- Conv/Deconv 权重布局不同：
  - Conv2d: PT `[O,I,H,W]` vs TF `[H,W,I,O]`
  - ConvTranspose2d: PT `[I,O,H,W]` vs TF `[H,W,O,I]`
- BatchNorm 细节：
  - epsilon 默认不同（PT `1e-5`，TF 常为 `1e-3`）
  - momentum 语义不同（更新公式方向相反）
  - 推理对齐测试应使用 eval/inference 模式并拷贝 running stats
- 插值差异：
  - PT `Upsample(..., align_corners=True)` 与 TF 默认 resize 行为不同
- `F.interpolate` 默认模式：
  - PT 默认 nearest，TF `tf.image.resize` 默认 bilinear（需显式指定）
- 复数算子：
  - `torch.fft` 与 `tf.signal` API 对齐需要验证 dtype/轴/shape
- 深度可分组卷积：
  - PT `groups=in_channels` vs TF `DepthwiseConv2D`

## 4) 迁移优先级与策略
- P0（先做，低风险高复用）：
  - NeuralILT 的 U-Net 主干（可覆盖大量 Conv/BN/Upsample 细节）
  - CFNO 核心频域层（覆盖复数运算、FFT、patch 变换）
- P1：
  - CFNOILT / CFNOLitho 全网络
  - DOINN 的 RFNO 层
- P2：
  - GANOPC / DAMOILT / LithoGAN / DAMOLitho（含判别器与对抗训练流程）
- 每一步必须有轻量对齐测试：
  - 小尺寸随机输入
  - PyTorch 权重拷贝到 TF 后比对前向输出（MAE/MAX）

## 5) 当前迁移进度
- 已新增 TF 代码：
  - `lithobench_tf/neuralilt_tf.py`：`UNetTF`
  - `lithobench_tf/cfnoilt_tf.py`：`CFNOTF`（CFNO 核心层）与 `CFNOILTNetTF`（整网）
  - `lithobench_tf/doinn_tf.py`：`RFNOTF`（RFNO 核心层）与 `RFNONetTF`（整网）
  - `lithobench_tf/ganopc_tf.py`：`GANOPCGeneratorTF` 与 `GANOPCDiscriminatorTF`
  - `lithobench_tf/lithogan_tf.py`：`LithoGANGeneratorTF` 与 `LithoGANDiscriminatorTF`
  - `lithobench_tf/damoilt_tf.py`：`DAMOILTGeneratorTF` 与 `DAMOILTDiscriminatorTF`
  - `lithobench_tf/damolitho_tf.py`：`DAMOLithoGeneratorTF` 与 `DAMOLithoDiscriminatorTF`
  - `lithobench_tf/cfnolitho_tf.py`：`CFNOLithoNetTF`
  - `lithobench_tf/__init__.py`
- 已新增轻量对齐测试：
  - `scripts/tf_port_smoke_tests.py`
  - 测试项：
    - `NeuralILT UNet` 前向对齐
    - `CFNO core` 前向对齐
    - `CFNOILT net tiny`（缩小配置整网）前向对齐
    - `RFNO core` 前向对齐
    - `RFNONet tiny`（缩小配置整网）前向对齐
    - `GANOPC generator` 前向对齐
    - `GANOPC discriminator` 前向对齐
    - `LithoGAN generator` 前向对齐
    - `LithoGAN discriminator` 前向对齐
    - `DAMOILT generator` 前向对齐
    - `DAMOILT discriminator` 前向对齐（务实阈值）
    - `DAMOLitho generator` 前向对齐
    - `DAMOLitho discriminator` 前向对齐
    - `CFNOLitho net tiny` 前向对齐
- 已新增训练路径烟测：
  - `scripts/tf_training_smoke_tests.py`
  - 覆盖各 TF 端口 1 step 前向+反向+优化器更新（含复数梯度检查）

## 6) 下一步执行清单（按顺序）
- [x] 跑通 `scripts/tf_port_smoke_tests.py` 并记录 MAE/MAX
- [x] 完整迁移 `CFNOILT`（整网）+ 小规模前向对齐
- [x] 完整迁移 `DOINN` 的 `RFNO/RFNONet` + 前向对齐
- [x] 迁移 `GANOPC` 生成器/判别器 + 前向对齐测试
- [x] 迁移 `LithoGAN`（共享模块复用）+ 双输出对齐
- [x] 迁移 `DAMOILT` + 前向对齐测试
- [x] 迁移 `DAMOLitho` + 前向对齐测试
- [x] 迁移 `CFNOLitho` + 前向对齐测试
