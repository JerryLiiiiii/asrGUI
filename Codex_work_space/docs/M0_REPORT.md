# asrGUI M0 技术验证报告

> 日期：2026-10-26
>
> 依据：[DESIGN_v0.1.md](../../Assets/doc/DESIGN_v0.1.md)、
> [ROADMAP_v0.1.md](../../Assets/doc/ROADMAP_v0.1.md)
>
> 结论：**GO，M0 技术验证完成**

## 1. 验证环境

| 项目 | 实测值 |
| --- | --- |
| Mac | MacBook Pro Mac15,6 |
| SoC | Apple M3 Pro，11 核 |
| 统一内存 | 18 GB |
| macOS | 27.0.1 |
| Python | Homebrew Python 3.11.17 |
| 执行设备 | CPU，4 线程 |
| 开发环境 | `Codex_work_space/.venv` |
| 数据与模型根 | `AppFile/runtime` |

PyTorch 2.11.0 编译包含 MPS，但当前系统返回
`torch.backends.mps.is_available() == False`，因此 M0 基线锁定 CPU。

## 2. 锁定基线

完整 Python 依赖见 [requirements.lock](../requirements.lock)，模型内容清单见
[models.lock.json](../models.lock.json)。关键版本为：

- FunASR 1.4.16
- ModelScope 1.40.2
- PyTorch / torchaudio 2.11.0
- PySide6 Essentials / Addons 6.12.0
- PyInstaller 6.22.3
- soundfile 0.14.0
- FFmpeg / ffprobe 9.0 Apple Silicon 静态构建

四个模型均声明 Apache-2.0。ModelScope 只提供可移动的 `master` 分支，因此
M0 的有效锁是 `models.lock.json` 中的内容 SHA-256，而不是分支名。

| 环节 | 模型 | 文件总量 | 主权重 SHA-256 |
| --- | --- | ---: | --- |
| ASR | SeACo Paraformer | 998,940,750 B | `3d491689244ec5dfbf9170ef3827c358aa10f1f20e42a7c59e15e688647946d1` |
| VAD | FSMN-VAD | 4,033,603 B | `b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5` |
| 标点 | CT-Punc large | 1,186,775,720 B | `7176cae922a872e130e6b88aef9a1153581711baf79c9124c7c95be383cd6f81` |
| 声纹 | CAM++ | 28,857,970 B | `3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8` |

## 3. 离线转写结果

所有测试均使用本地模型路径、`MODELSCOPE_OFFLINE=1`、
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并把 HTTP 代理指向不可用端口。

### 3.1 5 分钟样音

- 原始音频：MP3，48 kHz，双声道，320.095 秒。
- 分析音频：16 kHz，单声道，PCM signed 16-bit WAV。
- 结果：117 个有效句段、1665 个 token、2 名说话人。
- 峰值 RSS：4,376,952,832 B（约 4.08 GiB）。
- 包内 worker 实测：模型加载 17.924 秒，分块处理 38.485 秒，
  冷启动总耗时 56.409 秒，RTF 0.1202。

5 分钟样音使用 300 秒分块，因此实际处理为 2 块；最后 20 秒单独成块，
与不分块时的句段数量会有小幅差异。

### 3.2 30 分钟样音

- 原始音频：M4A/ALAC，48 kHz，双声道，1811.541 秒。
- 分块：7 块，每块 300 秒；最后约 11.5 秒。
- 结果：739 个有效句段、9481 个 token；自动输出 8 个说话人簇，人工确认实际为
  4 名说话人，详见第 4 节。
- 模型加载：18.724 秒。
- 全部块处理：256.224 秒。
- 冷启动总耗时：274.948 秒（约 4 分 35 秒）。
- RTF：0.1414。
- 峰值 RSS：4,616,290,304 B（约 4.30 GiB）。
- 该峰值低于设计文档的 10 GB 门槛。

分块 PCM 每次最多读取约 19 MiB；不会一次形成 2 小时整段 PCM。
输出通过校验：句段按时间递增，token ID 全会议唯一，token 时间单调且均落在
所属句段内，文本可逐 token 重建。

### 3.3 时间戳与标点结论

30 分钟结果包含 9481 个 token，其中 8651 个可点击：

- 4528 个 token 使用模型精确字/词时间。
- 4123 个 token 因源时间戳数量与字符数不一致，按句段范围插值。
- 830 个是标点或空白，不可点击，时间绑定到前一个可点击 token 终点。

结论：可以取得可用句子时间与大量真实 token 时间；原模型存在时间戳覆盖不完整，
但已验证“精确 / 插值 / 句级”三级精度标记和单调修正方案。

FunASR 1.4.16 还会在录音尾部生成只有标点的伪句。适配层会丢弃这类句段并记录
`dropped_non_text_segments`，不把它们当成有效转写。

## 4. 说话人接口

- CAM++ 嵌入维度为 192。
- 5 分钟样音内得到 2 个声学簇；6 个检查片段的同类平均余弦相似度 0.4062，
  异类平均 0.0826。
- 30 分钟样音跨 7 块合并后自动得到 8 个全局簇；经人工确认，样音实际为
  **4 名说话人**。
- FunASR 自带 `ClusterBackend` 在少于 20 个嵌入时直接返回全零，导致 30 分钟
  样本错误合并为 1 人；适配层改为以 CAM++ 中心做 cosine average-linkage
  聚类，阈值暂定 0.78。

结论分为两层：

1. CAM++ 嵌入提取、会议内聚类和余弦匹配接口可运行，人工确认建档路径技术可行，
   满足 M0 的接口验证目标。
2. 当前自动人数和聚类质量不合格。实际 4 人被识别为 8 组，属于过分割；
   其中包含多个小簇，且同类最小余弦 0.213 与异类最大 0.1789 很接近。

只调整当前 19 个局部中心的合并阈值无法解决问题：阈值降到 0.50 仍得到 5 组，
并保留单点小簇。M3 必须改为基于全部有效片段的全局聚类，结合静音感知切块、
人数范围、人工合并和未知回退。当前 8 组结果不得用于声纹建档或真实成员命名。

## 5. 检查点、取消与恢复

- 30 分钟任务在完成第 1/7 块后取消，检查点状态为 `cancelled`。
- 已完成的块以原子 JSON 提交；未完成块不标记成功。
- 续跑会跳过已校验块，只处理缺失块；取消点限定在安全块边界。
- 全量完成后检查点状态为 `completed`。

M0 只验证块间安全取消，不承诺在 FunASR 单块推理内部立即中止。M2 需要细化
阶段进度和取消反馈。

## 6. 最小桌面与打包

- PySide6 UI 通过 `QProcess` 启动独立 `asr-worker`，仅用 stdout JSON Lines
  通信。
- Bundle Identifier 为 `com.jerryliiiiii.asrgui`。
- PyInstaller onedir 应用内包含私有 Python、PySide6、FunASR、PyTorch、
  四个模型和静态 FFmpeg/ffprobe。
- 应用逻辑体积 3,890,607,749 B（约 3.62 GiB）。
- 在精简 `PATH=/usr/bin:/bin`、无效网络代理下，UI → worker 冒烟通过。
- 同一个包内 worker 实际完成 5 分钟离线 ASR，而不是只返回 mock 结果。
- 包内 FFmpeg 与 ffprobe 均可独立执行，`otool -L` 仅引用 Apple 系统框架/库。
- Personal Team 的 `Apple Development` 身份已生效并通过严格深签：
  - Identity SHA-1: `F6226ECC92F035B47A6E5D9F900CB62E1FFD2B45`
  - TeamIdentifier: `AU5X2T7C97`
  - `codesign --verify --deep --strict` 通过

已处理的冻结兼容问题：

1. FunASR 注册模型时调用 `inspect.getsourcelines()`，PyInstaller 归档无源码；
   通过运行时钩子提供诊断元数据回退。
2. Paraformer CIF predictor 使用 `torch.jit.script`，必须携带 FunASR `.py`
   源文件；构建脚本现已自动复制源文件。
3. FunASR 在少于 20 个说话人嵌入时不做聚类；跨块聚类改由适配层完成。

## 7. 许可证与打包边界

详见 [THIRD_PARTY.md](../THIRD_PARTY.md)。M0 已记录 FunASR、ModelScope、
PyTorch、Qt/PySide6、FFmpeg、CPython、PyInstaller 和四个模型的许可证及随包
条件。

当前 FFmpeg 静态构建启用 GPL 组件。现阶段只用于个人本地验证；若未来分发，
必须补齐 GPL 源码/提供方式与通知。Qt 当前为动态链接，后续分发时仍需满足
LGPL 的替换和通知义务。

## 8. 出口判断

| M0 出口条件 | 状态 | 证据 |
| --- | --- | --- |
| 完全断网加载并转写两份样音 | 通过 | 两份样音均使用本地模型和无效代理完成 |
| 可用句子时间，token 时间或降级策略明确 | 通过 | 精确/插值/句级标记与自动校验 |
| 说话人嵌入与会议内聚类接口可运行 | 通过 | CAM++ 192 维嵌入和聚类接口可运行；实际 4 人被过分割为 8 组，准确率修正归入 M3 |
| 30 分钟音频内存受控 | 通过 | 峰值约 4.30 GiB，分块处理 |
| 最小应用不依赖系统 Python/FFmpeg 并返回结果 | 通过 | 精简 PATH 下包内真实 ASR |
| 模型、依赖、许可证和打包边界已记录 | 通过 | 锁文件、哈希和第三方记录 |
| Personal Team 本地签名后启动 | 通过 | Apple Development 深签通过，签名后 UI/worker 冒烟通过 |

技术结论为 **GO，M0 技术验证可以正式关闭**。

## 9. 遗留风险

- 尚无人工校对基准，未计算 CER/WER；当前只验证结构与时间定位能力。
- 已确认当前跨块说话人聚类发生实际过分割：真实 4 人得到 8 组；该缺陷必须在
  M3 入口前修正，不能用当前标签建立声纹。
- 约 48% 可点击 token 为插值定位，后续需通过播放误差抽样确认约 300 ms 目标。
- CT-Punc large 主权重约 1.19 GB，对包体积贡献显著，后续可比较较小的
  `ct-punc-c` 候选。
- PyInstaller 包携带了 FunASR 的较宽模块面，当前包偏大，M5 前应做依赖裁剪。
- 尚未在第二台未安装开发工具链的 Mac 上验收；当前通过精简 PATH 和包内二进制
  模拟了独立性，但仍属于同机测试。

## 10. 实验产物

- 短音与说话人接口：`AppFile/runtime/m0/reference_rec_5min__c7e50cac`
- 30 分钟干净基准：`AppFile/runtime/m0/reference_rec_30min_clean`
- 包内真实短音验证：`AppFile/runtime/m0/packaged_5min_final`
- Personal Team 签名应用：`Codex_work_space/dist/asrGUI.app`

最终 Personal Team 签名 `.app`、模型和上述 M0 证据均保留。