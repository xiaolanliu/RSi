# 仿真参考示范

`sim_fold_clothes_01/success.mp4` 和 `sim_fold_clothes_02/success.mp4`
是实际 RoboDojo rollout 的三视角录像，
依次为顶视、左腕、右腕。模型为官方 RoboDojo 微调 pi05（checkpoint
59,999），不是主闭环默认的原始 `pi05_base`。

两条轨迹均在布局组 0 通过原生成功检查：

| 示范 | 布局 | 控制步数 | 仿真时间 | 视觉检查 |
|---|---:|---:|---:|---|
| 01 | 0 | 286 | 11.44 秒 | 最终衣物有明显褶皱 |
| 02 | 1 | 315 | 12.60 秒 | 袖子与下摆向内折叠，作为默认参考 |

全程由 VLA 执行，没有 GPT、mock 或人工接管。
`provenance.json` 记录权重身份、生成配置、原生结果和视频 SHA256。

成功标签指原生位置关系与回位判据。
两条视频供没有原始人工成功示教时测试上下文接口，不代表人工示教质量或总体成功率。
后续可用用户的早期示教成功 MP4 替换配置中的 `gpt.demo_mp4`。

`configs/fold_clothes_context.toml` 已引用示范 02，控制模型仍为原始 base，
GPT 默认关闭。安装和采集复现见 [操作说明](../../docs/AGENT_LOOP.md)。
