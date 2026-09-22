# 选择服务器与共用范围

本地项目维护一套技能和通用源码。连接、部署路径和验证级别由 [servers.json](../../../config/servers.json) 区分；运行时的 CAN、相机、关节边界等由每个现场的 primitive 配置区分。配置档案不保存密码、密钥或认证文件。

档案格式保留 `schema_version`，选择器读取任意数量的 `sites` 条目；新增设备按 [扩展接口](extending.md) 添加配置或驱动，保留现有控制契约。

| 服务器 ID | IP | 设备档案 | 接入状态 |
|---|---|---|---|
| agilex56 | 10.169.21.56 | [AgileX 四臂现场](site-agilex56.md) | 独立 primitive 已部署，有双臂运动与 RGBD 历史实测 |
| panfeng38 | 10.169.21.38 | [Panfeng Piper 现场](site-panfeng38.md) | Piper X；全部 8 个入口、CAN/关节使能、双臂上抬约 20 cm、RGBD 已接入/实测 |
| houzhi1 | 10.169.21.1 | [第三套平台](site-houzhi1.md) | 全部 8 个入口已部署；双臂同步上抬约 20 cm、夹爪和 RGBD 有实测 |

## 选择目标

优先使用用户本轮明确的 IP/ID；同一连续任务可沿用已经明确的目标。换服务器时重新读取对应档案和实际主机身份；不能沿用另一台服务器的动作目标、标定、相机序列号或部署路径。未指定且上下文不能唯一确定时，只询问目标服务器；不要默认 `.56`、广播到两台或在连接失败时自动换机。

在包含本项目源码的机器上，以下命令只显示档案，不建立连接或发送动作：

```bash
PYTHONPATH=src python3 -m agilex_control.sites list
PYTHONPATH=src python3 -m agilex_control.sites show --site 10.169.21.38
PYTHONPATH=src python3 -m agilex_control.sites show --site agilex56
PYTHONPATH=src python3 -m agilex_control.sites show --site houzhi1
```

从项目根目录运行，或使用绝对 `PYTHONPATH` 和 `--registry` 路径。输出的 `ssh_argv` 保留域用户名反斜杠；不要把 `SENSETIME\panfeng1` 当成 shell 转义后重新拆分。`primitive_deployment: null` 表示共享执行器尚未部署，不能回退到另一站点的 `config/site.json`。

SSH 连接后，核对 `hostname`、当前账户、项目和实际依赖；档案是已记录的信息，不是实时健康证明。部署记录中的 `supported_primitives`（若有）限定本站已接入的子集；同时读取 validation，接口部署不表示运动已验证。`houzhi1` 全部入口已部署，双臂同步上抬、夹爪恢复与 RGBD 已实测；实时状态仍重新读取。按选定现场的文档调用工具；一次调用只操作该服务器。`move_both` 是同一台服务器上的双臂共享时钟，不提供跨服务器同步。`stop` 也有服务器范围；全局停止请求应针对本任务实际启动过的所有服务器分别停止并检查反馈。

## 数据归属和部署

- 共用：观察—规划—动作—验证流程、单位约定、通用协议/运动学实现，以及经过条件核对的操作经验。
- 分开：SSH 账户、Python/项目路径、CAN 物理映射和主从拓扑、相机 SDK 序列号/角色、FK 与关节边界、标定、实测记录和运行目录。
- 相机参数、机械臂几何/零点、工具/负载、保护设置、场景目标和当前状态的独立范围见 [复用边界](portability.md)。参数缺失不跨设备补用；新配置/标定独立保存，不覆盖原文件。
- 新证据使用 `evidence/<日期>/host-<IP>/<任务>/`，派生标定必须记录服务器、臂和相机身份。旧 `experiments/hd_local_translation_model.json` 仅属于 `.56` 右臂对应现场。
- `config/site.json` 保留为 `.56` 的现有运行配置。`.38` 当前使用独立 `config/sites/panfeng38/control-piperx-20260913.json`，保存本站 CAN、相机、限位和 Piper X 几何；原 state/observe 和早期普通 Piper 候选配置保留为历史。手眼/TCP 标定仍未完成。
- 技能在本地阅读后经 SSH 调用远端工具即可；远端不要求安装技能副本，已有副本也不表示自动同步。当前共享技能更新在本地项目，远端副本的更新或部署应按任务需要进行，不混入现场密码/认证文件。
- 部署代码和文档不等于获得实机动作验证。保留当前会话的已有授权和最新停止指令，不能从别台设备或历史记录推导本次动作目标。
