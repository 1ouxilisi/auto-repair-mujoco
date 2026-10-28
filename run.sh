#!/bin/bash
# 一键跑通 auto-repair-mujoco 全流程
# 在 WSL 里执行：bash run.sh

set -e

echo "=== [1/4] 采集演示数据（50 条轨迹） ==="
python collect_demo.py --num_episodes 50 --output data/

echo ""
echo "=== [2/4] 训练行为克隆策略（200 epochs） ==="
python train.py --data data/auto_repair_demo.npz --epochs 200

echo ""
echo "=== [3/4] Rollout 验证 ==="
python rollout.py --checkpoint checkpoints/bc_mlp.pt

echo ""
echo "=== 完成 ==="
echo "如果看到 ✅，说明第一个技能包 baseline 跑通了。"
echo "下一步：把 scene.xml 里的任务加多步（断电→验电→...）。"
