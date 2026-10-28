# Changelog

本项目所有重要版本变更都记录在此。

## [Unreleased]

### 计划中
- V4：UR5 / Franka 真机模型 + Sim2Real 迁移
- V5：技能包打包 + 上架机器人应用商店

---

## [V3.0.0] - 2026-10-28

### Added
- Diffusion Policy 简化版实现
- Transformer Baseline vs Diffusion Policy 对比实验
- 8阶段完整维修任务（同 V2）
- 对比可视化图表（轨迹对比 + 误差柱状图）

### Changed
- 优化训练管线，支持多条演示数据
- 加标准化预处理，提升训练稳定性

### Notes
- 在小数据集（5条演示）+ CPU 训练条件下，Transformer baseline 效果优于简化版 Diffusion Policy
- Diffusion Policy 需要更多数据、更大模型才能发挥优势

---

## [V2.0.0] - 2026-10-28

### Added
- 8阶段完整电池包检查流程（移动→下降→找验电笔→拿笔→检测→收回）
- 条件 Transformer 模型（3层 decoder，d_model=64）
- 4合1 可视化图（完整轨迹 + 关节角度 + 单步误差 + 分阶段误差）
- 平均轨迹误差 + 最终位置误差双指标评估

### Results
- 最终位置误差：8.9mm
- 平均轨迹误差：2.7mm

---

## [V1.0.0] - 2026-10-28

### Added
- 4阶段维修任务（移动→下降→拿扳手→举起）
- Transformer Decoder 轨迹预测模型
- 轨迹可视化（末端轨迹图 + 关节角度图）
- 从起始观测预测整条动作轨迹

### Results
- 最终位置误差：1.2mm
- 轨迹长度：101步

---

## [V0.1.0] - 2026-10-28

### Added
- 最简 2-link 平面机械臂正运动学
- MLP 行为克隆 baseline
- 纯 PyTorch demo（不需要 MuJoCo）
- 完整管线：采集数据 → 训练 → Rollout 验证

### Results
- 最终位置误差：1.0mm
- 纯 CPU 可跑，零依赖

---

## [V0.0.1] - 2026-10-28

### Added
- 项目骨架初始化
- MuJoCo 场景文件（工作台 + 电池包 + 工具 + 6-DOF 机械臂）
- 数据采集脚本
- 训练脚本
- Rollout 验证脚本
- MIT License
