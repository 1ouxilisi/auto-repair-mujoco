# auto-repair-mujoco

> 汽车后市场机器人技能包的第一个开源仿真环境。
>
> 一个人，不买真机，在 MuJoCo 里把"电池包高压断电"这套汽车维修流程训成机器人策略。

## 这是什么

一个最简版汽车维修工位 MuJoCo 场景：工作台、绿色高压电池包、红色绝缘扳手、6-DOF 机械臂。
任务：让机械臂从 home 位置移动到电池包上方。

这是「具身智能一人公司」的第一个产品——不是论文，不是 Demo 视频，是一个能跑、能训、能复现的仿真环境。

## 为什么做这个

机器人公司（宇树、智元、Figure、Tesla）不懂汽车三电维修流程。
汽车后市场（4S 店、维修厂、电池回收）不知道机器人能干什么。

这个 repo 就是中间那座桥：**把汽车维修流程翻译成机器人能学的仿真任务**。

## 环境要求

- Python 3.11+（Windows / Linux / WSL 都行）
- MuJoCo 3.x
- PyTorch 2.x（CUDA 可选，CPU 也能跑）

## 快速开始

### 纯 PyTorch Demo（零依赖，直接跑）

```bash
# 不需要 MuJoCo，纯数学公式模拟机械臂正运动学
python demo_torch.py
```

预期输出：
```
✅ 成功！机械臂到达电池包上方，误差 1.0 mm
   pipeline 跑通：采集数据 → MLP 行为克隆 → 到达目标
```

### 完整 MuJoCo 流程

```bash
# 1. 采集 50 条演示轨迹
python collect_demo.py --num_episodes 50 --output data/

# 2. 训练行为克隆策略（MLP baseline，300 epochs）
python train.py --data data/auto_repair_demo.npz --epochs 300

# 3. Rollout 验证：看机械臂能不能到达电池包上方
python rollout.py --checkpoint checkpoints/bc_mlp.pt
```

预期输出：
```
最终到目标距离：0.0XX 米
✅ 成功！机械臂到达电池包上方。
```

## 项目结构

```
auto-repair-mujoco/
├── scene.xml           # MuJoCo 场景（工作台/电池包/工具/6-DOF机械臂）
├── collect_demo.py     # 采集演示数据（关键帧插值）
├── train.py            # 行为克隆训练（MLP baseline）
├── rollout.py         # 验证策略在仿真里跑起来
├── demo_torch.py      # 纯 PyTorch demo（不需要 MuJoCo，直接跑）
├── test_mujoco.py      # 快速验证 MuJoCo 环境是否正常
└── data/               # 采集的演示数据（gitignore）
```

## 当前状态

| 模块 | 状态 |
|---|---|
| 纯 PyTorch demo（2-link MLP） | ✅ 跑通，误差 1mm |
| 6-DOF MuJoCo 场景 | ✅ 加载成功 |
| 数据采集（50 episodes × 101 步） | ✅ |
| MLP 行为克隆训练（300 epoch） | ✅ loss 0.63→0.36 |
| Rollout 验证 | ⏳ 待 MuJoCo 兼容性修复 |

## 下一步路线

- [x] V0：最简场景 + MLP baseline，跑通"机械臂到电池包上方"
- [ ] V1：换 Diffusion Policy，复现 PushT 那套管线
- [ ] V2：加任务步骤（断电→验电→拆护板→...），变成多步骤技能
- [ ] V3：换成 UR5 / Franka 真实机械臂模型，Sim2Real 迁移
- [ ] V4：打包成技能包，等宇树 UniStore / 智元灵渠 OS 开放任务型应用那天上架

## 作者

一个人玩具身智能。不买真机、不报班、不考 ROS 证。
在扣子上做了个「具身智能技术雷达」智能体，专门回答 VLA / Sim2Real / 数据 / 本体格局的问题。
