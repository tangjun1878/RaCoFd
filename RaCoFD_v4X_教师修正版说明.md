# RaCoFD 专业检测教师修正版（RT-DETRv4-X）

## 改动与目的

本次修正原补丁默认使用 RT-DETRv2-X 的问题。新的专业检测教师为 **RT-DETRv4-X（HGNetv2-B5）**，学生为 **RT-DETRv4-S**，另一位独立教师仍是 **DINOv3-B**。

- `engine/RaCoFD/detection_teacher.py`：改为加载 RT-DETRv4-X 检测网络，使用已训练 X checkpoint；仅返回 `pred_logits`、`pred_boxes`，按需返回 `feature_f5`。
- `configs/racofd/racofd_hgnetv2_s_coco.yml`：教师配置改为 `configs/rtv4/rtv4_hgnetv2_x_coco.yml`，权重本地路径示例为 `pretrain/rtv4_hgnetv2_x_coco.pth`。
- `engine/RaCoFD/criterion.py`：支持 `reliability_filtering` 配置开关。
- `configs/racofd/racofd_hgnetv2_s_coco_naive_dual.yml`：关闭质量筛选，成为真正的朴素双教师消融。
- `engine/core/yaml_utils.py`：修复 `load_config(file_path, cfg=dict())` 的可变默认参数，避免同进程先后加载学生/教师 YAML 时产生交叉污染。

**注意：** RT-DETRv4-X 的 COCO 检测权重与 DINOv3-B 的权重是两个独立文件。专业检测教师在学生训练时只加载 RT-DETRv4-X 检测器；不会再次实例化 RT-DETRv4-X 训练时用过的 DINOv3 辅助教师。

## 配置文件用途

| 配置 | 说明 | 与完整方法的差异 |
| --- | --- | --- |
| `racofd_hgnetv2_s_coco.yml` | 完整 RaCoFD | 保留 DINOv3 DSI/GAM、X 分类+Box KD、GT-RoI VFM、稳定性/梯度证据、互补增强 |
| `racofd_hgnetv2_s_coco_1gpu.yml` | 单卡 | 只改 batch、workers、学习率，方法逻辑与主配置相同 |
| `racofd_hgnetv2_s_coco_naive_dual.yml` | 真正的朴素双教师消融 | 关闭分类/Box 可靠性筛选、实例 VFM、翻转稳定性和梯度证据；DINOv3 基础 DSI 与 X 分类/Box KD 保留 |
| `racofd_hgnetv2_s_coco_reliable_dual.yml` | 可靠双教师消融 | 可靠分类/Box + GT-RoI VFM，关闭互补调节（`complement_alpha: 0`） |
| `racofd_hgnetv2_s_coco_xfeat.yml` | X 特征蒸馏扩展 | 在完整方法基础上，增加 RT-DETRv4-X F5 空间能量特征蒸馏 |

## 启用方法

从项目根目录启动：

```bash
python train.py -c configs/racofd/racofd_hgnetv2_s_coco.yml -d cuda --use-amp --seed 42
```

单卡：

```bash
python train.py -c configs/racofd/racofd_hgnetv2_s_coco_1gpu.yml -d cuda --use-amp --seed 42
```

1. 下载 **官方 RT-DETRv4-X / HGNetv2-B5 的 COCO 检测 checkpoint**，存为 `pretrain/rtv4_hgnetv2_x_coco.pth`，或把 YAML 中 `checkpoint_path` 改为真实路径；不要使用 RT-DETRv2-X 权重。
2. 保持 DINOv3-B 权重配置有效：`pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`。
3. 准备 COCO 数据并检查配置中的数据目录。
4. 检测教师权重参数匹配覆盖率不足 95% 时会报错，不会自动降级为随机教师。

## 研究上的一个重要变化

RT-DETRv4-X 在其官方训练过程中已经使用 DINOv3 特征蒸馏，因此与独立 DINOv3-B 教师的知识可能存在更多重叠。推荐把“VFM + RT-DETRv4-X”作为主设定，再与“VFM + RT-DETRv2-X”做教师来源消融，以验证知识重叠对互补蒸馏的影响（在研究需要时）。

## 验证范围

本补丁在当前环境完成了 Python 静态编译、YAML 解析与差异检查；由于缺少真实 RT-DETRv4-X、DINOv3 权重和 COCO 数据集，尚未进行真实 GPU 教师前向或端到端训练。正式训练请先在目标环境进行若干 batch 的前向与梯度检查。
