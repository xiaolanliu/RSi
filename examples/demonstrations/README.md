# 仿真参考示范

`sim_fold_clothes_01/success.mp4` 是实际 RoboDojo rollout 的三视角录像，
依次为顶视、左腕、右腕。模型为官方 RoboDojo 微调 pi05（checkpoint
59,999），不是主闭环默认的原始 `pi05_base`。

本条轨迹在布局组 0、布局 0，经过 286 个原生控制步（11.44 秒）通过
仿真成功检查。全程由 VLA 执行，没有 GPT、mock 或人工接管。
`provenance.json` 记录权重身份、生成配置、原生结果和视频 SHA256。

视频最终衣物仍有明显褶皱；成功标签仅指原生位置关系与回位判据。
它供没有原始人工成功示教时测试上下文接口，不代表人工示教质量或总体成功率。
后续可用用户的早期示教成功 MP4 替换配置中的 `gpt.demo_mp4`。

`configs/fold_clothes_context.toml` 已引用此视频，控制模型仍为原始 base，
GPT 默认关闭。安装和采集复现见 [操作说明](../../docs/AGENT_LOOP.md)。
