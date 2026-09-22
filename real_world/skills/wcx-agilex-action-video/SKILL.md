---
name: wcx-agilex-action-video
description: 为 AgileX 按执行阶段录制三路真实视频，去除推理等待，在回放顶部展示阶段行动与理由摘要并导出上下拼接；用于自动留存动作、带说明的回放和复用录像服务。
---

# 按动作阶段录制真实视频

复用项目中的 [recording.py](../../src/agilex_control/recording.py)、[phase.py](../../src/agilex_control/phase.py) 和 [导出脚本](../../scripts/export_action_video.py)。机器人接入与动作仍走 [操控 skill](../wcx-agilex-control/SKILL.md)，本技能不另建运动控制器或相机采集器。

## 启动与调用

先按本次明确的服务器核对相机键、相机服务 runtime、Python 和部署源码。相机由原 cameras 服务独占 USB；新增录像服务通过它的 `preview` RPC 读取 JPEG，录像进程不打开 CAN 或 USB。只有 `cameras.py` 支持 preview 且 `phase.py` 接入 recording 时才能直接启用；其他现场不能仅凭本地代码就宣称已支持。

在选定服务器项目根目录运行一个独立进程，可用现有用户 systemd transient unit 管理。命令中的目录必须来自本次现场与任务：

```bash
PYTHONPATH=src python -m agilex_control.recording serve \
  --runtime /tmp/wcx_agilex_recording \
  --camera-runtime /tmp/wcx_agilex_control \
  --output-root /absolute/task/video-clips --fps 15 --max-seconds 180
```

给本次控制配置的独立副本增加：

```json
{"action_recording":{"runtime_dir":"/tmp/wcx_agilex_recording"}}
```

让本次 client registry 指向这个配置，再照常调用 `client call` / `client phase`。含动作的阶段会在持有原运动控制权后启动录像，确认首帧写入才执行，并在退出时收尾；只观察、只取状态和已开始阶段的重复调用不新建动作片段。直接调用 primitive CLI 不经过 phase，因而不会自动录制。新动作使用新阶段身份，传输重试保持旧身份和目录，不为补录像重放动作。

`python -m agilex_control.recording status --runtime ...` 返回服务和当前片段状态。必要时独立调用 `start/stop --identity ...` 包围一段已授权执行；stop 需要匹配身份。服务同一时刻只录一段，已用身份不能覆盖。

## 阶段行动说明

需要在视频顶部解释“准备做什么、为什么这样做”时，使用 phase 的已有 context 字段，在执行前填写：

```json
{
  "steps": [],
  "context": {
    "video": {
      "title": "搭建底层",
      "action": "把已夹稳的长积木放到底层预定位置。",
      "reason": "先形成连续支撑，再放上层方块。"
    }
  }
}
```

上例只展示字段结构，实际 steps 应填入根据当前观测规划好的动作；空 steps 不触发录像。每段用一个短标题及简短的行动、依据摘要；三个字段均为1–300字符，通常各一句即可。依据来自当前可见情况与任务目标，不放内部逐步推演日志，也不提前把预期结果说成成功。若动作失败，保留当时计划，下一阶段写明依据新观测采取的调整。

新 recording 服务将摘要保存到 clip.json 的 stage 字段并回传；客户端确认摘要已保存才开始该段动作。升级后在空闲时重启录像进程，旧服务未保留摘要时本段不执行。单动作若需要说明，也用只有一步的 phase；传输重试继续保持原请求和阶段身份。

若需要把最后一次只读稳定性检查录入视频，可用现有 `recording.request(runtime, {"op":"start","identity":identity,"stage":summary})` 显式启动片段，立即调用原observe，并在finally中以相同identity发送stop。把这段顺序调用放在一次工具编排中，避免启动后夹入长时间推理；无需为录像发送空move。CLI的独立start没有stage参数，有摘要时使用上述现有RPC。092城墙验收按此方式录得71帧真实视频。

导出脚本自动在三路画面上方增加 ASTRA 标题、行动和依据区域，按每段精确帧数切换，原相机画面不被文字遮盖。摘要同时进入 manifest。全部旧片段都无 stage 时维持旧版布局；混合输入中缺少摘要的片段明确显示未提供，不事后编造理由。渲染需要 Pillow 和可显示中文的字体，自动查找已知 macOS/Noto 字体，也可用 --font 指定。字幕卡是说明图形，真实运动仍来自录制的 MP4。

## 时间与故障含义

片段覆盖阶段运行时间，包括规划、运动、停稳和收尾观察；模型在两次工具调用之间的推理不会录进去。它不是仅保留关节速度非零的剪辑。每路保存 MP4、原相机时间戳/帧号的 `frames.jsonl` 和 `clip.json`。相机之间没有硬件同步。

配置 FPS 是输出恒定帧率。读取迟到时会保持上一帧补足时间，补帧数和最大采样间隙明确记录；不能把 15 FPS 输出宣称为每秒 15 张新的原生图像。不要用事后观测图片补成缺失的运动。

要求录制的阶段若启动失败，会在动作前失败。动作失败或正常停止仍收尾保存已有片段；录像收尾状态独立于动作成败。出现 `finalization_uncertain` 时检查原片段与服务，不重放动作。进程被强杀无法保证 finally 执行，服务仍受 `max-seconds` 限制；`duration_limit` 表示录像截断，不表示机械臂停止。阶段长度与上限应匹配，异常退出后核对服务是否仍录制。

## 导出与验证

取回已经结束的 `video-clips` 目录。保留实际失败和纠正动作；若排除无动作试录，显式列出其身份。原视频直接拼接，默认上前视、中左腕、下右腕；其他相机键通过 `--cameras` 按这个顺序传入。

```bash
python scripts/export_action_video.py /absolute/video-clips /absolute/new-export \
  --exclude-identity recording-smoke-test --ffmpeg /absolute/ffmpeg
```

输出三路单独 H.264 MP4、`stacked.mp4` 和时间对应 `manifest.json`；有阶段说明时另含 `stage-header.mp4`。输出目录必须新建，避免覆盖。每路跨片段的相机序列号与 FPS 必须一致；同一实体相机在已结束片段之间更改分辨率时，按该路最大宽高等比例适配、居中补黑边，保留完整画面。原生尺寸逐片段写入 manifest，不能用其他相机替换缺失视角。完整解码四个输出（带说明为五个），核对总帧数/时长，并查看阶段边界、开始、中间、结束画面及最后实际任务结果。视频导出成功不等于机器人任务成功。

相机模式更改需要在录像空闲时独立进行，保留旧配置并记录新配置、服务名称和新内参。读取每帧 metadata，不把旧分辨率的主点用于新图像。2026-09-15 城墙任务将 `.56` 右腕由640×480增宽到848×480；三段640→848→640合成片段验证22帧全部保留、100×100标记尺寸不变、窄帧两侧各104像素黑边，六个字幕边界均匹配。序列号/FPS变化被拒绝。此项为离线导出验证，实机动作结果独立验收。

原生录像当前为MPEG4编码，不能将不同尺寸的MPEG4片段直接交给同一个concat解码器；实录导出曾在增宽处产生宏块错误，虽然输出仍可播放，画面已损坏。现在按连续相同尺寸分组独立解码、归一到相同H.264画布，再按原顺序流拷贝拼接；FFmpeg以-xerror在解码错误时失败。最初只有H.264的合成验证未覆盖此问题。补充的MPEG4三尺寸22帧试验及三段实录888帧均验证五个输出完整，888帧右腕输出逐帧与等比例补边后的原片段比较，最高平均像素差约0.703/255。错误成片单独标记拒收，原始录像不改，也不为重导出重放动作。

默认使用 libx264 软件编码。macOS 上若 FFmpeg 提供且已用短片验证 VideoToolbox，可显式加 `--encoder h264_videotoolbox` 加速，当前硬件分支为6Mbps；编码器写入 manifest。不因硬件编码失败重放机器人动作；用新导出目录重试离线编码。原片段保留在服务器，可在原片段所在机器已有 FFmpeg 的情况下就地导出，再取回结果，以减少传输。

2026-09-14 首次在 `.56` 部署并实录；[本次证据](../../evidence/2026-09-14/host-10.169.21.56/write-ace-evening/) 保存试录、真实阶段、源码修改前副本和离线验证。其他两站尚未因本次工作自动部署或验证。

2026-09-14 城墙任务进一步实测阶段摘要。`context.video` 经 phase、录制服务、clip 元数据、导出 manifest 保留；三段实录的27.533秒导出已逐帧核对字幕切换，五个输出各413帧完整解码。真实空抓阶段同样保留，下一阶段摘要解释重新观察和增大开口的原因，不能在导出时把原计划改写成成功。证据见[城墙任务](../../evidence/2026-09-14/host-10.169.21.56/castle-wall-evening/README.md)。

城墙最终87段回放共26717帧、1781.133秒。五个输出以FFmpeg -xerror完整解码通过，174个字幕首末边界匹配源卡片，并抽查拼接画面中的阶段对应与混合尺寸过渡；这才将完整成片标为已验证。见[最终导出与验收](../../exports/2026-09-14/host-10.169.21.56/castle-wall-annotated-action-video/README.md)。
