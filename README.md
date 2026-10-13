# RM 视觉第二周作业 — 装甲板识别与 6D 位姿解算

C++17 + CMake + OpenCV + ONNX Runtime 的**双方案**实现：

| 方案 | 流程 | 入口 |
|---|---|---|
| 传统 | 颜色差分找灯条 → 几何约束配对 → mlp 数字网络确认 | `--method trad` |
| 神经网络 | Infantry-v8n 关键点回归 → NMS 去重 | `--method nn` |

两种方案输出统一的 `Armor`（四角点契约序：左上/右上/右下/左下），
共用同一套 PnP 位姿解算、可视化与真值对比代码。

## 目录结构

    include/  8 个头文件（types / camera / pnp_solver / number_classifier /
              traditional_detector / nn_detector / truth_loader / visualizer）
    src/      对应 8 个 .cpp + main.cpp
    models/   mlp.onnx  Infantry-v8n.onnx  label.txt        （素材, 不入库）
    data/     测试视频                                      （素材, 不入库）
    images/   测试图 img1~3.jpg（960x720）                   （素材, 不入库）
    truth/    真值 img1~3.md                                （素材, 不入库）
    results/  程序输出（结果图 / 结果视频）

## 依赖

| 依赖 | 版本（本机实测） |
|---|---|
| Ubuntu (WSL2) | 26.04 |
| g++ | 15.2.0（C++17） |
| CMake | ≥ 3.16 |
| OpenCV | 4.10.0 |
| ONNX Runtime | 1.20.1（解压于 `~/onnxruntime`） |

## 构建

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## 运行

```bash
# 图片 · 传统方案 + 真值对比（仿真图必须用 camera_sim.yaml, 颜色随该文件为 blue）
./build/rm_vision images/img1.jpg --method trad --truth truth/img1.md --camera camera_sim.yaml

# 图片 · 神经网络方案 + 真值对比
./build/rm_vision images/img1.jpg --method nn --truth truth/img1.md --camera camera_sim.yaml

# 视频 · 传统方案（实拍视频为橙红灯条, 用 camera.yaml 默认值）
./build/rm_vision data/test.avi --method trad --output results/video_trad.mp4

# 视频 · 神经网络方案
./build/rm_vision data/test.avi --method nn --output results/video_nn.mp4
```

参数说明：

| 参数 | 说明 | 默认 |
|---|---|---|
| `<路径>` / `--source` | 输入图片或视频（必填） | — |
| `--method` | `trad` / `nn` | `trad` |
| `--truth` | 真值 md（图片模式可选，用于误差对比） | 空 |
| `--camera` | 相机参数 yaml | `camera.yaml` |
| `--models` | 模型目录 | `models` |
| `--output` | 输出图片/视频路径（父目录自动创建） | 空 |
| `--enemy` | `red` / `blue` 覆盖 yaml 中的敌方颜色 | yaml 值 |
| `--show` | 窗口显示（Esc 退出） | 关 |
| `--step` | 视频抽帧步长 | 1（逐帧） |

## 实现要点

- 工程带**两套相机参数**：`camera.yaml` 用于实拍视频（1440x1080，重投影 0.1448px）；
  `camera_sim.yaml` 用于仿真图片（960x720，零畸变，重投影 0.0140px，`camera_matrix`/`dist_coeffs`
  与战队给出的 `rune_sim_camera` 数据逐位一致）——
  **两者是完全独立的两台相机，不可互相缩放混用**；跑图片时用 `--camera camera_sim.yaml`。
- 程序按 `实际宽度 / image_width` 适配同一相机的其他分辨率（fx/fy/cx/cy 等比缩放，畸变不变）；
  若输入宽高比与标定不一致，会打印警告并**拒绝**套用错误的内参（而非静默给出错值）。
- 畸变不手动去；直接作为参数传入 `cv::solvePnP`，由其内部处理。
- PnP 采用 `SOLVEPNP_IPPE`（平面四点目标专用），并做 tz>0 与重投影 RMS 自检。
- 外法线定义 = `-R` 的第三列（板坐标系 +z 背离相机）；真值对比时用法线夹角评估姿态精度。
- 素材（models / data / images / truth）与构建产物均不入库，见 `.gitignore`。
