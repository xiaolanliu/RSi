# 本次发布验收

- `environment.json`：新建隔离虚拟环境的版本和硬件；没有继承原项目的 site-packages。
- `verification_cpu.json`、`verification_cuda.json`：完整8条在线 + 1条离线轨迹，共13548帧。在线两种设备共用历史参考；离线CPU的独立原代码参考出处见 `provenance/offline_cpu_reference.json`。
- `unit_tests.txt`：36项机制测试。
- `preparation_smoke.json`：3条真实 LeRobot episode，经路径映射、离线教师与正常训练归一化，共3688帧。其中1164帧教师输出对比独立原报告。
- `refit_smoke.json`：35条历史轨迹各截64帧，测试显式路径的正常参考拟合→动作单元→加权校准→独立包加载，共2240帧。仅接口/数据流测试，不是完整重训或检测性能评估。
- `browser.json`、`replay_preview.png`：Chromium网页检查与回放预览。

本次没有重新跑全量2000轮训练。发布推理权重和原始两份2000轮checkpoint的SHA256不变。原完整数据集的历史统计保存在 `provenance/`，不与这里的小样例测试混为一谈。
