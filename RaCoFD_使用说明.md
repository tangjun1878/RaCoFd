# RaCoFD：基于用户上传 RT-DETRv4 源码的完整修改

> 项目：RaCoFD (Reliability-Aware Complementary Foundation–Detector Distillation)。
> 所有改动基于 `RT-DETRv4_pack_1116.zip` 的实际代码接口。
> 本补丁不包含 COCO 数据集、DINOv3 权重或 RT-DETRv2-X 权重。

## 1. 网络角色

- **通用特征教师**：沿用 RT-DETRv4 的 `DINOv3TeacherModel`，冻结，只输出 F5 对应的 768 通道特征，用于原 DSI/GAM 密集蒸馏和新增 GT 实例特征蒸馏。
- **专业检测教师**：`RaCoFDDetectionTeacher`，使用 `configs/rtv2/rtv2_r101vd_6x_coco.yml` 的 RT-DETRv2/R101-vd 大模型，加载真实训练好的 COCO checkpoint；冻结，至少蒸馏**分类响应和边界框**，可选蒸馏 F5 空间能量特征。
- **学生**：原 `RTv4` 的 HGNetv2-S + HybridEncoder + D-FINE Decoder，不改变实际检测输出或推理图。训练时从 HybridEncoder 暴露共享 F5 和 DINO 对齐投影。

## 2. 方法损失

1. **独立 VFM 密集特征 KD**：RT-DETRv4 现有 `loss_distill` + GAM，原样保留。
2. **GT 匹配的检测响应 KD**：教师和学生分别对相同 GT 执行 Hungarian Matching；按 GT ID 建立预测对应，不按 Query 数组下标对齐。逐类别 Bernoulli KL 对应 Sigmoid 分类，边界框 SmoothL1 + GIoU。分类可靠性以 GT 类别正确性及置信度衡量；定位可靠性以与 GT 的 IoU 衡量；分类与 Box 分别加权。
3. **DINOv3 实例特征 KD**：依据 GT 框分别 RoIAlign 教师与学生投影 F5，并计算余弦距离；不依赖大检测教师框来决定 GT 区域。
4. **证据约束互补**：冻结 DINOv3 的原图与水平镜像图特征在还原镜像后衡量区域稳定性；按间隔采样主检测分类/定位损失与实例 VFM 损失在**学生共享 F5** 上的局部梯度余弦；结合教师的分类/定位可靠性得到互补增强权重。权重下界为 1，故 X 教师错误不会直接关闭独立基础知识。
5. **可选 X 特征 KD**：通过三尺度检测特征中的 F5 空间能量图匹配避免跨通道直接模仿。开启时必须同时将专业教师 `return_features` 设为 `True`。

## 3. 文件变更

新建：

- `engine/RaCoFD/__init__.py`
- `engine/RaCoFD/detection_teacher.py`
- `engine/RaCoFD/criterion.py`
- `configs/racofd/racofd_hgnetv2_s_coco.yml`
- `configs/racofd/racofd_hgnetv2_s_coco_1gpu.yml`
- `configs/racofd/racofd_hgnetv2_s_coco_naive_dual.yml`
- `configs/racofd/racofd_hgnetv2_s_coco_reliable_dual.yml`
- `configs/racofd/racofd_hgnetv2_s_coco_xfeat.yml`

修改原工程（必须覆盖）：

- `engine/__init__.py`：注册 RaCoFD Criterion。
- `engine/core/_config.py`：新增检测教师属性。
- `engine/core/yaml_config.py`：读取 `teacher_detect_model` 配置。
- `engine/solver/_solver.py`：训练时加载两位冻结教师；测试只加载学生；不保存教师权重到 checkpoint。
- `engine/solver/det_solver.py`：将检测教师传入每轮训练。
- `engine/solver/det_engine.py`：执行两教师前向和可选 DINO 镜像视图，向 Criterion 传递结果。
- `engine/rtv4/hybrid_encoder.py`：训练期暴露共享 F5（用于有效性证据）。
- `engine/rtv4/rtv4.py`：训练输出增加共享 F5/检测 F5；推理结构不变。
- `engine/rtv4/rtv4_criterion.py`：修复原工程中无蒸馏特征的辅助头直接创建 CUDA 零张量的兼容问题。

## 4. 安装与准备

在原项目根目录执行（有网络的环境）：

```bash
pip install -r requirements.txt
```

准备三个必要资源：

1. COCO 2017 数据集，按原 `configs/dataset/coco_detection.yml` 配置路径。
2. DINOv3-B 权重，原配置为 `pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`；保持原 `dinov3/` 源码目录。
3. **已在 COCO 上训练的 RT-DETRv2-X / ResNet101-vd 检测权重**，默认配置 `pretrain/rtdetrv2_r101vd_coco.pth`。必须与 `configs/rtv2/rtv2_r101vd_6x_coco.yml` 对应；不匹配将明确报错，不会悄悄使用随机权重。权重文件须来自可信来源。

此外，`HGNetv2` Backbone 需要相应预训练资源，或通过合适的 student tuning 权重初始化。

## 5. 启动训练

**主配置：**

```bash
python train.py -c configs/racofd/racofd_hgnetv2_s_coco.yml -d cuda --use-amp --seed 42 -t ""
```

**单卡配置：**

```bash
python train.py -c configs/racofd/racofd_hgnetv2_s_coco_1gpu.yml -d cuda --use-amp --seed 42 -t ""
```

这里 `-t ""` 用于覆盖原项目 `train.py` 的非空默认 tuning 路径，表示不加载完整学生权重。如需要从指定 student 权重开始，请改成 `-t /path/to/your/student_checkpoint.pth`，并确保所有对照采用公平的初始化和训练预算。

如实际文件路径不同，调整 `configs/racofd/racofd_hgnetv2_s_coco.yml` 中的两位教师权重路径，或使用原项目的 `-u` 配置覆盖机制。

**消融配置：**

```bash
python train.py -c configs/racofd/racofd_hgnetv2_s_coco_naive_dual.yml -d cuda --use-amp -t ""
python train.py -c configs/racofd/racofd_hgnetv2_s_coco_reliable_dual.yml -d cuda --use-amp -t ""
python train.py -c configs/racofd/racofd_hgnetv2_s_coco_xfeat.yml -d cuda --use-amp -t ""
```

**仅评估学生：**

```bash
python train.py -c configs/racofd/racofd_hgnetv2_s_coco.yml -d cuda --test-only -r /path/to/last.pth -t ""
```

训练以外的测试/推理不加载 DINOv3 或 RT-DETRv2-X 教师。

## 6. 训练成本及局限

- 镜像一致性默认额外运行一次冻结 DINOv3；资源紧张可以设 `flip_consistency: False`，此时不再使用双视图表征稳定性证据。
- 任务梯度一致性默认每 4 步估计一次，其他步使用按类别的指数平滑先验；额外 autograd probe 带来训练时开销，但不参与部署。
- X 教师的分类与定位质量使用有标签训练集 GT 判断，不能用于无标注数据的直接照搬。
- 模型采用共享 F5 上的局部梯度一致性作为**任务有效性的代理证据**，不是教师知识正确性的数学证明；其实际有效性需要通过 COCO 及跨数据集实验验证。
- RT-DETRv2-X 的 EMA 权重、输入标签映射和归一化与训练配置必须一致；当前默认面向 COCO 80 类。
- 此项目没有附带真实数据和两位教师的完整权重，因此本包只在合成数据上完成端到端烟雾测试；**不能声称已跑通真实 COCO 训练或得到论文 AP 结果**。

## 7. 已进行的代码测试

- 真实学生 RT-DETRv4-S 实例化、128×128 合成图像前向、Criterion 与反向传播；
- 新 Criterion 单元测试：常规 GT、空 GT、错误分类教师、检测特征 KD、梯度证据计算；
- RT-DETRv2 教师加载测试：使用测试用 R18 配置及匹配的临时随机测试 checkpoint，验证独立加载、动态尺寸和前向输出格式；该随机权重仅用于代码测试，不用于训练或结果声明；
- 真实 `train_one_epoch` 调用：使用合成样本及两位模拟教师完成单次优化器更新；
- `configs/racofd/` 主配置与单 GPU 配置注册、模型构造、Criterion 注入；
- 所有改动文件均通过 Python 编译检查。

**建议正式跑 COCO 时先做 1–2 个 batch 的训练检查，再正式训练和消融。**
