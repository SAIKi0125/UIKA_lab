# 鸟类降落轨迹生成模块 — 设计方案

- 日期: 2026-05-31
- 项目: himloco_lab
- 任务: 在 Isaac Sim / Isaac Lab 中训练四足机器人完成"接鸟"任务，需要一个能批量生成"鸟降落轨迹"的模块。

---

## 1. 目标

为 RL 环境每个 episode 提供一条 (T, ≥6) 维 `(pos, vel)` 时间序列，作为一个 kinematic 小球（代表鸟）的逐步姿态。

- **形状/速度** 接近真实飞行下滑数据分布
- **风速** 与轨迹一致地随机化
- **落点** 在机狗周围一个圆内随机
- 不需要外部条件输入（不需要指定起点/落点/风速给生成器；这些由调用方在外层平移/旋转处理）

---

## 2. 数据来源

ArduPilot dataflash log，路径 `/home/esd/LOGS/0000000{1..9}.BIN`，共 9 个文件，约 70 MB。

确认可用消息（已用 pymavlink 验证 `00000004.BIN`）：

| 消息 | 字段 | 用途 |
|---|---|---|
| `XKF1` | `PN/PE/PD`(NED 位置), `VN/VE/VD`(NED 速度), `Roll/Pitch/Yaw` | 主轨迹源 |
| `DCM` | `VWN/VWE/VWD` | 三维风速估计 |
| `ARSP` | `Airspeed` | 空速（备用） |
| `POS` | `Lat/Lng/Alt` | 备用绝对位置 |

时间戳 `TimeUS` 微秒。XKF1/DCM 在 4 号日志中均为 4290 条，约 50 Hz。

---

## 3. 模块定位与边界

- 训练数据是**飞行体（无人机）下滑**，不是真实鸟。本模块把它当作"近似鸟降落形态"使用，不追求生物学真实。
- 模块**只负责生成轨迹**，不负责：
  - 鸟（小球）的资产创建（由 RL env 用 `RigidObjectCfg` 自建）
  - 接触判定、奖励、机狗策略
- 模块输出由 Isaac Lab 的 `CommandTerm` 接口暴露给 RL 环境。

---

## 4. 总体架构

```
mdp/bird/
  parser.py        # ArduPilot BIN -> raw (t, pos_NED, vel_NED, wind_NED, airspeed)
  segment.py       # 自动检测"下滑+减速"段
  dataset.py       # 重采样到固定 dt + 平移到落点为原点 + 旋转水平接近方向到 +X
  vae.py           # VAE 模型 (encoder/decoder)
  train.py         # 离线训练脚本，输出 vae.pt + norm_stats.json
  generator.py     # 推理时批量解码 + 预生成池
  command.py       # BirdTrajectoryCommand（CommandTerm）
  command_cfg.py   # BirdTrajectoryCommandCfg
  assets/          # 训练好的权重 + 归一化统计
```

两阶段使用：

- **离线**：跑一次 `train.py`，从 BIN 训练 VAE，存权重和归一化统计
- **在线**：RL 启动时 `BirdTrajectoryCommand` 加载权重 → 一次性 decode N=1e5 条轨迹缓存到 GPU 池 → 每次 env reset 从池里随机索引一条 → 每个 sim step 按 episode 进度插值，写到一个 observation/term，并驱动 RL env 里那个 kinematic 小球

---

## 5. 数据处理细节

### 5.1 切分（segment.py）
状态机扫 XKF1 时间序列：
- **起点**：连续 N 帧 `VD > 0.5 m/s` 且高度累计下降 > Δh
- **终点**：水平速度 `||VN, VE|| < v_min` 或 `alt < 阈值`
- **长度过滤**：保留 [3 s, 10 s] 范围内的段

预期产出：5–9 个 BIN 文件总计若干十到几百条段。先做一次预处理后查看分布，不足则做增广。

### 5.2 归一化（dataset.py）
每条轨迹独立做：
1. **平移**：以末端位置为新原点
2. **旋转**：把水平接近方向（末端前 K 帧的平均水平速度方向）旋到 +X 轴，消除 yaw
3. **重采样**：插值到固定 `dt=20 ms`、`T=250` 步（5 s）。短的左侧补 nan 后掩码丢弃，长的截尾
4. **风速同步变换**：水平风按 yaw 旋转，垂直风原样
5. **统计归一化**：保存全局 mean/std 到 `norm_stats.json`，运行时反归一化

每条样本最终维度：`(T=250, D=6)` = `(px, py, pz, vx, vy, vz)`，配套一个 3 维风向量 `(VWN', VWE', VWD)` 在旋转后的坐标系下。

### 5.3 增广（可选）
- 随机 yaw 旋转（0–360°）
- 时间轻微缩放 ±10%
- 位置/速度叠加小高斯噪声

---

## 6. 模型

**VAE**（不是 CVAE，因为不需要外部条件）：
- 输入：把轨迹 `(T, D)` 与风速 `(3,)` 联合成 `(T, D+3)` 后展平为 1D
- Encoder/Decoder 先 baseline 用 **MLP**（hidden=256/512、latent=16）
- 训练目标：标准 ELBO（重建 MSE + KL），可加 β-VAE 退火避免坍缩
- 改进路线：跑通后换 **1D-CNN**

输出维度规划：解码出 `(T, D+3)` 即 `(traj, wind)` 联合，保证轨迹与风速的相关性来自数据。

---

## 7. 推理路径

### 7.1 预生成池（GPU）
- 启动时 `decoder(z ~ N(0, I))` 一次跑 1e4–1e5 条
- 反归一化后得到 `(N, T, 6)` 轨迹和 `(N, 3)` 风
- 全部缓存在 GPU

### 7.2 BirdTrajectoryCommand
- `_resample_command(env_ids)`（episode reset 时调用）：
  - 从池里给每个 env 随机抽一个索引
  - 从机狗周围 `R=[r_min, r_max]` 圆内随机采一个落点 xy（z 取地面高度或固定）
  - 随机一个全局 yaw 旋转角，把池中"已对齐到 +X"的轨迹旋到这个朝向
  - 缓存这个 env 的 `(T, 6)` 全程轨迹 + 起始 sim step
- `_update_command()`（每 sim step 调用）：
  - 用 `(current_step - start_step) * dt` 在 `(T, 6)` 上线性插值得到当前 `(pos, vel)`
  - `write_root_pose_to_sim` / `write_root_velocity_to_sim` 写到鸟（小球）
  - 把 `(相对机狗位置, 相对速度)` 暴露给 observation

### 7.3 时间对齐
- 轨迹总时长 5 s，episode 长度需 ≥ 5 s
- 鸟从 episode 开始的某个 t0 出发（可固定也可随机），中段触地，给机狗预测+移动时间

---

## 8. 与 RL 环境的接口

- `BirdTrajectoryCommandCfg` 字段：
  - `weights_path: str` — VAE 权重
  - `norm_stats_path: str` — 归一化统计
  - `pool_size: int` — 预生成池大小
  - `bird_asset_name: str` — RL env 中小球 asset 名
  - `target_radius: tuple[float, float]` — 落点半径范围
  - `target_height: float` — 落点 z
  - `traj_dt: float`, `traj_steps: int`
  - `start_offset_range: tuple[float, float]` — episode 起 t0 范围
- 复用 `mdp/commands/commands.py` 的注入风格，作为新的 `CommandTerm` 子类
- 暴露给 observation 的项：
  - 鸟相对机狗的 `(pos, vel)`（必有）
  - 鸟未来 N 帧的 `(pos, vel)` 预览（可选，做"先知"实验）

---

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 切分后样本太少 | 多 BIN 合并 + 增广（yaw/时间/噪声）；或回退到解析骨架 + 物理扰动混合 |
| 飞控轨迹与"鸟"形态差异大 | 第 1 步先可视化样本，肉眼判断是否符合直觉；不符合再调切分判据或换数据 |
| VAE 模式坍缩 | 监控重建误差 + latent 多样性；β-VAE、KL annealing；先 MLP 后 1D-CNN |
| GPU 内存 | 1e5 条 × 250 × 6 × 4 byte ≈ 600 MB，可承受；超了就缩 pool_size |
| 鸟与机狗碰撞导致小球被弹飞 | kinematic 小球忽略外力；接触只用于触发任务事件，不做物理交互 |

---

## 10. 推荐执行顺序

1. **先解析一个 BIN，可视化几条切出来的下滑段** — 验证数据质量与形态
2. 固化预处理脚本，生成数据集 npz
3. **MLP-VAE baseline 训练**，肉眼检查采样合理性
4. GPU 批量推理 + 预生成池
5. `BirdTrajectoryCommand` 接入 Isaac Lab，跑通鸟在仿真中飞动
6. 接 RL 环境，给机狗加 observation，开始训练接鸟策略

第 1 步是 go/no-go 关卡：如果可视化结果不像可用的鸟降落，则回到方案设计阶段重新讨论（解析+物理混合等替代路线）。

---

## 11. 已确认的设计决策

- 引擎：**VAE**（不是 CVAE，因不需外部条件）
- 数据来源：用户提供的 ArduPilot BIN
- 切分策略：自动检测下滑+减速段
- 风处理：与轨迹联合喂给 VAE 一起学
- 落点：episode reset 时在机狗周围圆内随机
- 鸟实体：kinematic 小球，由 BirdTrajectoryCommand 每步 set_pose/set_vel
- 模块路径：`source/himloco_lab/himloco_lab/tasks/locomotion/mdp/bird/`

---

## 12. 未决 / 后续可议

- 鸟在落点之后是否做"反弹/滚动"等表演（任务设计范畴）
- 是否给机狗暴露"未来 N 帧预览"作为先知特征
- 课程学习：是否随训练进度扩大落点半径、加大风速
- 当前未保存代码，仅保留本设计文档；后续可再次进入 brainstorming 流程进入实现阶段
