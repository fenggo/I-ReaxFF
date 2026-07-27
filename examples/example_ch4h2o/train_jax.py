#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JAX 版 ReaxFF 训练脚本, 对应原 train.py (irff.reax_nn, TensorFlow backend).
使用 irff.reaxff_jax (JAX backend) 替代.

用法:
    python train_jax.py                          # 仅前向计算 + 损失评估
    python train_jax.py --train                  # 训练 (optax + jax.grad)
    python train_jax.py --train --lr 0.001 --step 5000
"""
import os, sys
import json as js
import numpy as np
import jax
import jax.numpy as jnp
import optax
from irff.reaxff_jax import ReaxFF_nn
from irff.data.ColData import ColData

# ── 数据加载 ──────────────────────────────────────────────────────
getdata = ColData()

dataset = {'h22-v': 'aimd_h22/h22-v.traj'}

strucs = ['h2o2', 'ch4w2', 'h2o16']

weight_energy = {'h2o2': 2.0, 'others': 2.0}
weight_force  = {'h2o16-0': 0, 'ch4w2-0': 1}

batchs = {'others': 10000}

for mol in strucs:
    b = batchs.get(mol, batchs['others'])
    trajs = getdata(label=mol, batch=b)
    dataset.update(trajs)

# ── 参数约束 ──────────────────────────────────────────────────────
clip = {
    'Desi':   (100.0, 725.0),
    'bo1':    (-0.3, -0.002),  'bo2': (4.0, 9.9),
    'bo3':    (-0.3, -0.003),  'bo4': (4.0, 9.9),
    'bo5':    (-0.3, -0.003),  'bo6': (4.0, 9.9),
    'rosi':   (0.5, 1.5),  'ropi': (0.5, 1.46),  'ropp': (0.5, 1.46),
    'ovun1':  (0.0, 1.0),   'ovun2': (-1.20, 0.0),
    'ovun3':  (0.0066, 6.0), 'ovun4': (0.0, 36.0), 'ovun5': (0.0, 85.0),
    'pen1':   (-23.9, 27.4),
    'rvdw_C': (1.84, 2.399), 'rvdw_O': (1.84, 2.50), 'rvdw_H': (1.62, 2.39),
    'rvdw_N': (1.9, 2.79),   'rvdw_H-N': (1.65, 2.4), 'rvdw_H-O': (1.64, 2.79),
    'rvdw_C-H': (1.64, 2.38),
    'Dehb':   (-3.998, 0.0), 'Dehb_C-H-O': (-3.9, -0.35),
    'rohb':   (1.877, 2.392), 'hb1': (2.72, 3.64), 'hb2': (18.7, 19.64),
    'Devdw':  (0.001, 0.8),  'alfa': (6.0, 17.0),
    'vdw1':   (0.50, 8.0),
    'gammaw': (1.7, 14.0),
    'val1':   (10, 60),       'val2': (0.05, 1.98),
    'val3':   (0.01, 7.6),    'val4': (0.01, 0.698),
    'tor1':   (-5.0, -0.049), 'tor2': (0.41, 5.0),
    'tor3':   (0.041, 5.0),   'tor4': (0.05, 1.0),
    'V1':     (-10.0, 24),    'V2': (0, 48), 'V3': (0.0, 10),
    'cutoff': (0.0001, 0.01), 'acut': (0.0010, 0.010),
}

cons = [
    'lp2', 'lp1',
    'theta0', 'val8',
    'valang', 'val9',
    'valboc', 'cot1', 'cot2', 'coa1', 'coa2', 'coa3', 'coa4',
    'val', 'lp3',
    'theta0', 'vale', 'val9', 'val10', 'val8', 'valang',
]

mf_layer = [9, 2]

# ── 构建模型 ──────────────────────────────────────────────────────
print("Building ReaxFF_nn model (JAX backend)...")
rn = ReaxFF_nn(
    dataset=dataset,
    libfile='ffield.json',
    MessageFunction=3,
    mf_layer=mf_layer,
    be_layer=[9, 1],
    cons=cons,
    clip=clip,
    weight_force=weight_force,
    weight_energy=weight_energy,
    screen=True,
    fixrcbo=False,
    lambda_bd=30000.0,
    lambda_reg=0.001,
)

print(f"  Structures: {rn.strcs}")
print(f"  Total frames: {rn.nframe}")
print(f"  Trainable params: {len(rn.pp)}")
for k in sorted(rn.pp.keys()):
    v = rn.pp[k]
    print(f"    {k}: {np.array(v).shape}")


# ── 前向计算 + 损失评估 ──────────────────────────────────────────
def forward_and_loss():
    """对所有结构做一次前向计算, 返回总损失."""
    total = jnp.array(0.0)
    loss_e = jnp.array(0.0)
    loss_f = jnp.array(0.0)
    for st in rn.strcs:
        rn.forward(st)
        total += rn.get_loss(st)
        loss_e += rn.loss_e
        loss_f += rn.loss_f
    return total, loss_e, loss_f


def evaluate():
    """评估并打印当前损失."""
    total, le, lf = forward_and_loss()
    print(f"  loss(total): {float(total):12.5f}  "
          f"energy: {float(le):12.5f}  force: {float(lf):12.5f}")
    return float(total)


# ── 训练: 有限差分法逐参数更新 ───────────────────────────────────
def train_finite_diff(step=500, lr=0.0001, eps=1e-5, print_step=10, writelib=500):
    """用有限差分法计算梯度, 逐参数更新 (适用于小规模参数).

    注意: 这是为 JAX 状态类设计的简单训练方法.
    对于大规模参数训练, 建议将 ReaxFF_nn 重构为纯函数后使用 jax.grad.
    """
    if not rn.pp:
        print("WARNING: no trainable parameters. Running forward-only.")
        return

    param_keys = list(rn.pp.keys())
    print(f"Training with finite-difference: {step} steps, "
          f"lr={lr}, {len(param_keys)} parameters")

    for i in range(step + 1):
        # 当前损失
        total_loss, _, _ = forward_and_loss()

        if i % print_step == 0:
            print(f"  {i:6d}  loss: {float(total_loss):12.5f}")

        if i == step:
            break

        # 对每个参数计算有限差分梯度
        for k in param_keys:
            old_val = rn.pp[k]
            p_val = jnp.array(old_val)

            # f(p + eps)
            rn.pp[k] = p_val + eps
            rn.p[k] = p_val + eps
            loss_plus, _, _ = forward_and_loss()

            # f(p - eps)
            rn.pp[k] = p_val - eps
            rn.p[k] = p_val - eps
            loss_minus, _, _ = forward_and_loss()

            # 梯度 = (f(p+eps) - f(p-eps)) / (2*eps)
            grad = (loss_plus - loss_minus) / (2.0 * eps)

            # 更新
            new_val = p_val - lr * grad
            rn.pp[k] = new_val
            rn.p[k] = new_val

        # 参数约束
        rn.clamp()

        if writelib > 0 and i > 0 and i % writelib == 0:
            fname = f'ffield_jax_{i}.json'
            rn.save_ffield(fname)
            print(f"  -> saved {fname}")

    rn.save_ffield('ffield_jax.json')
    print("Done. Final parameters saved to ffield_jax.json")


# ── 训练: 纯函数式 (推荐, 需要适配) ─────────────────────────────
def train_pure_function(step=1000, lr=0.0001, print_step=10, writelib=500):
    """
    纯函数式训练 (推荐方式).

    思路: 将整个 forward + loss 封装为纯函数, 然后用 jax.grad 求梯度.
    这要求 ReaxFF_nn 的方法是纯函数 (无副作用), 目前 reaxff_jax.py
    使用有状态类, 因此需要额外适配.

    如果 ReaxFF_nn 被重构为纯函数, 训练代码简化为:
        loss_fn = jax.jit(jax.value_and_grad(pure_forward_and_loss))
        for i in range(step):
            loss_val, grads = loss_fn(params, data)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
    """
    print("Pure-function training requires ReaxFF_nn to be refactored as pure functions.")
    print("Falling back to finite-difference training.")
    train_finite_diff(step=step, lr=lr, print_step=print_step, writelib=writelib)


# ── 主入口 ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='JAX ReaxFF Training')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--step', type=int, default=500, help='Training steps')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='Finite difference epsilon')
    parser.add_argument('--print', type=int, default=10, help='Print interval')
    parser.add_argument('--writelib', type=int, default=500,
                        help='Save ffield every N steps')
    parser.add_argument('--evaluate', action='store_true',
                        help='Only evaluate, no training')
    args = parser.parse_args()

    if args.evaluate:
        print("Evaluation mode:")
        evaluate()
    elif args.train:
        # 使用有限差分法训练 (适合参数较少的场景)
        train_finite_diff(
            step=args.step, lr=args.lr, eps=args.eps,
            print_step=args.print, writelib=args.writelib)
    else:
        # 默认: 评估一次
        print("Default: single evaluation (use --train for training)")
        evaluate()
        print("\nUsage:")
        print("  python train_jax.py --evaluate        # 仅评估")
        print("  python train_jax.py --train            # 训练 (有限差分)")
        print("  python train_jax.py --train --step 2000 --lr 0.001")