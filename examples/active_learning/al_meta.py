#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
active_learning.py — 分块式主动学习(Active Learning)循环脚本
=============================================================
针对分子晶体体系的 ReaxFF-nn 力场主动学习。

新流程 (分块 + restart):
 每一轮迭代:
    1. 1000步 MD 模拟 (NVT + coordNum CV + harmonicWalls)
       → 首个 chunk 从 data.lammps 启动，后续从 restart 续跑
       → 每个 chunk 结束写入 write_restart
    2. 调用 mlpkit.critical() 判断失稳结构
       → 无失稳: 用新 restart 继续下一个 1000 步
       → 有失稳: 进入 DFT + 训练流程
    3. DFT 计算 (lm.py, siesta 单点算能量/力, 打标签)
    4. 训练 (train.py --e=N, 更新 ffield.json)
    5. 力场同步: ffield.json → ffield, 拷贝到 META_DIR
    6. 回滚: 从失稳 chunk 之前的 restart 文件重新启动 MD
       (用新力场, 但偏置势状态继续累积)

用法:
    python active_learning.py                      # 运行 1 轮
    python active_learning.py --iters 5            # 运行 5 轮
    python active_learning.py --epochs 500         # 每轮训练 500 epoch
    python active_learning.py --chunk-size 2000    # 每 chunk 2000 步 (默认 1000)
    python active_learning.py --max-chunks 500     # 最大 chunk 数 (默认无限制)
    python active_learning.py --max-md-steps 1000000  # MD 总步数上限

依赖:
    - lammps (ReaxFF-nn + COLVARS)
    - mlpkit (Anaconda Python, 含 mlpkit.critical)
    - 工作目录: /home/feng/mlff/tnt/meta/ (MD) + /home/feng/mlff/tnt/ (训练)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import glob

import numpy as np

# =============================================================
#                  配置 (按实际环境修改)                        =
# =============================================================
META_DIR    = '/home/feng/mlff/tnt/meta'   # metaD 工作目录
TRAIN_DIR   = '/home/feng/mlff/tnt'        # 训练工作目录
LABEL       = 'ct4'
NPROCS      = 12

# Python 环境
ANACONDA_PY = '/home/feng/.local/anaconda/bin/python3'  # mlpkit 所在 Python
LOCAL_PY    = sys.executable                             # 当前 Python

# LAMMPS
LMP         = 'lammps'
MPIRUN      = 'mpirun'

# 数据文件
DATA_FILE   = os.path.join(META_DIR, 'data.lammps')

# ── 元素检测 (从 data.lammps 头注释行) ──
# LAMMPS data 文件顶部格式:
#   #/atom 1 carbon
#   #/atom 2 hydrogen
#   含义: type 1 = C, type 2 = H
# pair_coeff 顺序: C H N O (对应 type order)
_ELEM_NAME_MAP = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
                  6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
                  11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
                  16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
                  21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
                  26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn'}

# 对应原子量, 用于 Masses 段反推元素 (容差 ±0.5)
_ELEM_MASS = {1: 1.008, 2: 4.0026, 3: 6.94, 4: 9.0122, 5: 10.81,
              6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180,
              11: 22.990, 12: 24.305, 13: 26.982, 14: 28.085, 15: 30.974,
              16: 32.06, 17: 35.45, 18: 39.948, 19: 39.098, 20: 40.078,
              21: 44.956, 22: 47.867, 23: 50.942, 24: 51.996, 25: 54.938,
              26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.38}


def detect_elements(data_file=None):
    """从 data.lammps 文件头读取元素映射.

    检测顺序:
      1. #/atom 注释行 (irff/ReaxFF-nn 格式)
      2. Masses 段 (根据原子量反推元素, 容差 ±0.5 amu)

    Returns:
        elements: list of str, 按 type 顺序 (['C','H','N','O'])
        elem_str: 空格分隔字符串 ('C H N O'), 用于 pair_coeff
    """
    if data_file is None:
        data_file = DATA_FILE
    if not os.path.exists(data_file):
        return None, None

    elem_map = {}   # type_num -> symbol

    with open(data_file) as f:
        raw = f.read()

    # 方法 1: #/atom 注释行
    for line in raw.split('\n'):
        s = line.strip()
        if not s.startswith('#/atom '):
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                tnum = int(parts[1])
            except ValueError:
                continue
            name = parts[2].lower()
            if len(name) <= 2:
                symbol = name.capitalize()
            else:
                for z, sym in _ELEM_NAME_MAP.items():
                    if sym.lower() == name:
                        symbol = sym
                        break
                else:
                    symbol = name[:2].capitalize()
            elem_map[tnum] = symbol

    if not elem_map:
        # 方法 2: 从 Masses 段推断 (原子量 → 元素)
        print(f"    ℹ️  无 #/atom 注释, 尝试从 Masses 段推断...")
        in_masses = False
        mass_map = {}  # type_num -> mass
        for line in raw.split('\n'):
            s = line.strip().lower()
            if 'masses' in s:
                in_masses = True
                continue
            if in_masses:
                if not s:
                    continue   # 跳过空行
                if s.startswith('#') or 'atoms' in s or \
                   'bond' in s or 'angle' in s or 'dihedral' in s or \
                   'pair' in s or 'velocities' in s:
                    break
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        tnum = int(parts[0])
                        mass = float(parts[1])
                    except (ValueError, IndexError):
                        continue
                    mass_map[tnum] = mass

        if mass_map:
            for tnum, mass in sorted(mass_map.items()):
                # 最接近的原子量匹配
                best_z, best_diff = 0, float('inf')
                for z, sym in _ELEM_NAME_MAP.items():
                    ref_mass = _ELEM_MASS.get(z, 0)
                    if ref_mass == 0:
                        continue
                    diff = abs(mass - ref_mass)
                    if diff < best_diff and diff < 0.6:
                        best_diff = diff
                        best_z = z
                if best_z > 0:
                    elem_map[tnum] = _ELEM_NAME_MAP[best_z]

    if not elem_map:
        print(f"    ⚠️  无法从 {data_file} 检测元素, 请用 --elements 指定")
        return None, None

    elements = [elem_map[i] for i in sorted(elem_map.keys())]
    elem_str = ' '.join(elements)
    print(f"    🔬 检测元素: {elem_str}")
    return elements, elem_str

# =============================================================
# 文件路径
# =============================================================
META_IN_CHUNK = os.path.join(META_DIR, 'in.meta_chunk.lammps')
COLVARS       = os.path.join(META_DIR, 'colvars.meta_nvt')
FFIELD_META   = os.path.join(META_DIR, 'ffield')
FFIELD_JSON   = os.path.join(TRAIN_DIR, 'ffield.json')
TRAIN         = os.path.join(TRAIN_DIR, 'train.py')
LM_SCRIPT     = os.path.join(TRAIN_DIR, 'lm.py')

# Restart 命名: restart.chunk_<N>  (第 N 个 chunk 结束后的 restart)
# restart.chunk_0 = 初始态 (data.lammps 等效)
RESTART_PATTERN = os.path.join(META_DIR, 'restart.chunk_*')

DUMP_DIR    = os.path.join(META_DIR, 'chunks')  # 每个 chunk 的 dump 存这里

# =============================================================
# 工具函数
# =============================================================

def run_cmd(cmd, timeout=None, cwd=None, check=True, shell=False,
            logfile=None):
    """运行命令, 可选将输出写入 log 文件"""
    print(f"\n>>> {' '.join(cmd) if not shell else cmd}")
    t0 = time.time()
    try:
        if logfile:
            with open(logfile, 'w') as fh:
                proc = subprocess.run(cmd, cwd=cwd, shell=shell,
                                      stdout=fh, stderr=subprocess.STDOUT,
                                      text=True, timeout=timeout)
        else:
            proc = subprocess.run(cmd, cwd=cwd, shell=shell,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True, timeout=timeout)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"    [exit {proc.returncode}] ({dt:.0f}s)")
            if not logfile and proc.stdout:
                print(proc.stdout[-800:])
            if check:
                raise RuntimeError(
                    f"命令失败: {cmd[0]} (exit {proc.returncode})")
        else:
            print(f"    [ok] ({dt:.0f}s)")
        return proc
    except subprocess.TimeoutExpired:
        print(f"    [timeout after {timeout}s]")
        return None


def md_healthy(log):
    """检查 MD log: 是否崩溃"""
    if not os.path.exists(log):
        return True  # 文件不存在视为正常
    with open(log, errors='ignore') as f:
        txt = f.read()
    bad = ['ERROR', 'Lost atoms', 'Non-numeric', 'NaN',
           'not enough space', 'MPI_ABORT', 'simulation unstable']
    return not any(b in txt for b in bad)


# =============================================================
# 分块 MD
# =============================================================

def generate_chunk_input(chunk_id, restart_src, nsteps, elements=None):
    """生成 in.meta_chunk.lammps — 单 chunk 的 LAMMPS 输入
    
    Args:
        elements: 元素列表, 如 ['C','H','N','O']. None 则回退到默认 C H N O.
    """
    if elements is None:
        elements = ['C', 'H', 'N', 'O']
    elem_str = ' '.join(elements)
    
    lines = []
    lines.append(f"# Chunk {chunk_id}: {nsteps} steps")
    lines.append("")

    if restart_src is not None:
        lines.append(f"read_restart    {restart_src}")
    else:
        lines.append("units           real")
        lines.append("atom_style      charge")
        lines.append("atom_modify     map array")
        lines.append("")
        lines.append("read_data       data.lammps")
        lines.append("")
        lines.append("# 初始化速度 (每个迭代重新设种子)")
        lines.append(f"velocity        all create 300 {7789 + chunk_id}")
        lines.append("")

    lines.append("# ReaxFF-nn")
    lines.append("pair_style      reaxff control nn yes checkqeq yes")
    lines.append(f"pair_coeff      * * ffield {elem_str}")
    lines.append("")
    lines.append("neighbor        2.5  bin")
    lines.append("neigh_modify    every 1 delay 1 check no page 200000")
    lines.append("")
    lines.append("# COLVARS metaD")
    lines.append("fix             2 all colvars colvars.meta_nvt")
    lines.append("")
    lines.append("# NPT")
    lines.append("fix             1 all npt temp 350.000000 350.000000 100 iso 0.000000 0.000000 100")
    lines.append("fix             Q all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff")
    lines.append("")
    lines.append("thermo_style    custom step temp epair etotal press vol "
                "cella cellb cellc # f_2")
    lines.append("thermo          1")
    lines.append("")
    dump_file = os.path.join(DUMP_DIR, f"chunk_{chunk_id:04d}.lammpstrj")
    lines.append(f"dump            1 all custom 1 {dump_file} "
                 "id type xu yu zu fx fy fz")
    lines.append("dump_modify     1 sort id")
    lines.append(f"log             meta_npt_chunk_{chunk_id:04d}.log")
    lines.append("")
    lines.append("timestep        0.1")
    lines.append("")
    lines.append(f"run             {nsteps}")
    lines.append("")
    restart_out = f"restart.chunk_{chunk_id:04d}"
    lines.append(f"write_restart   {restart_out}")

    with open(META_IN_CHUNK, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return META_IN_CHUNK, dump_file, restart_out


def run_md_chunk(chunk_id, restart_src, nsteps, elements=None, timeout_s=36000):
    """运行一个 MD chunk.

    Args:
        chunk_id:    chunk 编号 (从 1 开始)
        restart_src: 上一个 restart 文件路径 (None 表示从 data.lammps 开始)
        nsteps:      本 chunk 步数
        elements:    元素列表, 如 ['C','H','N','O']
        timeout_s:   超时 (秒)

    Returns:
        (success: bool, dump_file: str, restart_out: str, log_file: str)
    """
    in_file, dump_file, restart_out = generate_chunk_input(
        chunk_id, restart_src, nsteps, elements)

    log_file = os.path.join(META_DIR, f"meta_npt_chunk_{chunk_id:04d}.log")

    # 删除旧的 dump 和 restart（如果同名存在）
    for f in [dump_file, restart_out, log_file]:
        if os.path.exists(f):
            os.remove(f)

    cmd = [MPIRUN, '-np', str(NPROCS), LMP, '-in', in_file]
    print(f"\n{'─'*50}")
    print(f"  Chunk {chunk_id}: {nsteps} 步, "
          f"restart={'初始' if restart_src is None else restart_src}")
    print(f"{'─'*50}")

    try:
        proc = subprocess.run(
            cmd, cwd=META_DIR,
            stdout=open(log_file, 'w'), stderr=subprocess.STDOUT,
            timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"    ⚠️  Chunk {chunk_id}: MD 超时")
        return False, dump_file, restart_out, log_file

    restart_path = os.path.join(META_DIR, restart_out)
    healthy = md_healthy(log_file)
    restart_ok = os.path.exists(restart_path)

    if proc.returncode != 0:
        print(f"    ⚠️  Chunk {chunk_id}: LAMMPS exit {proc.returncode}")
        return False, dump_file, restart_out, log_file

    if not restart_ok:
        print(f"    ⚠️  Chunk {chunk_id}: 未生成 restart 文件")
        return False, dump_file, restart_out, log_file

    if not healthy:
        print(f"    💥 Chunk {chunk_id}: MD 崩溃")
        return False, dump_file, restart_out, log_file

    print(f"    ✅ Chunk {chunk_id}: 完成 → {restart_out}")
    return True, dump_file, restart_out, log_file


# =============================================================
# mlpkit.critical 调用 (通过 Anaconda Python subprocess)
# =============================================================

CRITICAL_CALL = """\
from mlpkit.core import critical
critical(dump='{dump}', output='{output}', score_threshold={threshold},
         crash_score={crash}, baseline_frames={baseline},
         one_per_run={one_per}, min_persist={min_persist})
"""


def call_mlpkit_critical(dump_file, output='critical.traj',
                         score_threshold=3.0, crash_score=50.0,
                         baseline_frames=10, one_per_run=True,
                         min_persist=3):
    """调用 mlpkit.critical() 判断失稳结构.

    Returns:
        (has_critical: bool, critical_traj_path: str or None)
    """
    if not os.path.exists(dump_file):
        print(f"    ❌ dump 不存在: {dump_file}")
        return False, None

    critical_out = os.path.join(META_DIR, output)
    # 删除旧输出
    if os.path.exists(critical_out):
        os.remove(critical_out)

    code = CRITICAL_CALL.format(
        dump=dump_file.replace('\\', '\\\\'),
        output=critical_out.replace('\\', '\\\\'),
        threshold=score_threshold,
        crash=crash_score,
        baseline=baseline_frames,
        one_per=str(one_per_run),
        min_persist=min_persist,
    )

    print(f"    🔍 mlpkit.critical: {dump_file}")
    proc = subprocess.run(
        [ANACONDA_PY, '-c', code],
        cwd=META_DIR,
        capture_output=True, text=True, timeout=300)

    if proc.stdout:
        for line in proc.stdout.strip().split('\n'):
            print(f"       {line}")
    if proc.stderr:
        for line in proc.stderr.strip().split('\n'):
            print(f"       [stderr] {line}")

    if os.path.exists(critical_out):
        # 检查是否真的有帧
        from ase.io import read
        try:
            frames = read(critical_out, index=':')
            if len(frames) > 0:
                print(f"    ⚠️  发现 {len(frames)} 个失稳帧!")
                return True, critical_out
            else:
                print(f"    ✅ 无失稳帧")
                return False, None
        except Exception:
            # 可能为空文件，尝试用 ASE 判断
            fsize = os.path.getsize(critical_out)
            print(f"    critical.traj 存在但无法解析 (size={fsize})")
            return False, None
    else:
        print(f"    ✅ 无失稳帧")
        return False, None


# =============================================================
# DFT
# =============================================================

def run_dft(label='cb22', ncpu=None):
    """siesta DFT 单点: 运行 lm.py"""
    if ncpu is None:
        ncpu = NPROCS

    if not os.path.exists(LM_SCRIPT):
        print(f"    ❌ 找不到 {LM_SCRIPT}")
        return False

    cwd = os.getcwd()
    os.chdir(TRAIN_DIR)

    log_path = os.path.join(TRAIN_DIR, 'lm.log')
    proc = subprocess.run(
        [ANACONDA_PY, LM_SCRIPT],
        stdout=open(log_path, 'w'), stderr=subprocess.STDOUT,
        timeout=90000)

    os.chdir(cwd)

    out = os.path.join(TRAIN_DIR, f"{label}.traj")
    if proc.returncode != 0:
        print(f"    ❌ lm.py 失败 (exit {proc.returncode})")
        return False
    # if not os.path.exists(out):
    #     print(f"    ⚠️ DFT 未生成 {label}.traj")
    #     return False

    from ase.io import read
    labeled = read(out, index=':')
    print(f"    ✅ DFT 完成. ")
    return True


# =============================================================
# 训练 + 力场同步
# =============================================================

def run_training(epochs, timeout_s=8*3600):
    """训练 ReaxFF-nn"""
    run_cmd([ANACONDA_PY, TRAIN, f'--e={epochs}'],
            cwd=TRAIN_DIR, timeout=timeout_s, check=False)


def sync_ffield():
    """把训练产物 ffield.json 转成文本 ffield 并同步到 MD 工作目录"""
    # mlpkit.core.ffield 在 Anaconda Python 环境
    code = f"""\
import sys
sys.path.insert(0, '/home/xuni/mlpkit')
from mlpkit.core import ffield as mlpkit_ffield
mlpkit_ffield(jsonfile='{FFIELD_JSON}', ffieldfile='{os.path.join(TRAIN_DIR, "ffield")}')
print('ffield.json -> ffield OK')
"""
    proc = subprocess.run(
        [ANACONDA_PY, '-c', code],
        capture_output=True, text=True, timeout=60)
    print(proc.stdout.strip() if proc.stdout else "")
    if proc.stderr:
        print("   ", proc.stderr.strip()[-500:])

    src = os.path.join(TRAIN_DIR, 'ffield')
    if not os.path.exists(src):
        print("    ⚠️ 没有生成 ffield, 力场未同步")
        return False

    shutil.copy(src, FFIELD_META)
    print(f"    🔄 力场同步: {src} → {FFIELD_META}")
    return True


# =============================================================
# 主循环
# =============================================================

def main():
    ap = argparse.ArgumentParser(description='分块式主动学习循环: chunked MD → critical → DFT → 训练')
    ap.add_argument('--iters', type=int, default=1,help='主动学习轮数 (DFT+训练次数, 默认 1)')
    ap.add_argument('--epochs', type=int, default=300,help='每轮训练 epoch (默认 1000)')
    ap.add_argument('--chunk-size', type=int, default=1000,help='每 chunk 的 MD 步数 (默认 1000)')
    ap.add_argument('--max-chunks', type=int, default=None,help='最大 chunk 数 (默认无限制)')
    ap.add_argument('--max-md-steps', type=int, default=None,help='MD 总步数上限 (默认无限制)')
    ap.add_argument('--md-timeout', type=int, default=2*3600,help='每 chunk MD 超时秒数 (默认 2h)')
    ap.add_argument('--critical-threshold', type=float, default=3.0,help='mlpkit.critical score_threshold (默认 3.0)')
    ap.add_argument('--critical-crash', type=float, default=50.0,help='mlpkit.critical crash_score (默认 50.0)')
    ap.add_argument('--min-persist', type=int, default=3,help='mlpkit.critical 连续异常帧数 (默认 3)')
    ap.add_argument('--elements', type=str, default=None,
                    help='元素列表, 空格分隔, 如 "C H N O". '
                         '默认从 data.lammps 头注释行自动检测')
    ap.add_argument('--data-file', type=str, default=None,
                    help='data.lammps 路径 (默认 META_DIR/data.lammps)')
    args = ap.parse_args()

    # ── 元素检测 ──
    data_file = args.data_file or DATA_FILE
    elements = None
    if args.elements:
        # 用户显式指定, 最高优先级
        elements = args.elements.strip().split()
        print(f"    🧪 用户指定元素: {' '.join(elements)}")
    else:
        elems_detected, _ = detect_elements(data_file)
        if elems_detected:
            elements = elems_detected
        # else: elements 保持 None, generate_chunk_input 回退到 C H N O

    # ── 初始化 ──
    os.makedirs(DUMP_DIR, exist_ok=True)

    # 清理旧的 COLVARS 偏置态 (全新的主动学习周期)
    for f in ['out.colvars.state', 'out.colvars.state.old','out.colvars.traj', 'out.pmf', 'rest.colvars.state']:
        p = os.path.join(META_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"    🧹 清理: {f}")

    # 清理旧 chunk restart
    for old_rst in glob.glob(RESTART_PATTERN):
        os.remove(old_rst)
        print(f"    🧹 清理旧 restart: {old_rst}")

    # 清理旧 chunk 日志和 dump
    for old_log in glob.glob(os.path.join(META_DIR, 'meta_npt_chunk_*.log')):
        os.remove(old_log)
    for old_dump in glob.glob(os.path.join(DUMP_DIR, 'chunk_*.lammpstrj')):
        os.remove(old_dump)

    # ── 主循环 ──
    al_iteration = 0          # 已完成 DFT+训练 的次数
    chunk_id = 0              # 当前 chunk 编号
    total_md_steps = 0        # 累计 MD 步数
    total_chunks = 0          # 总 chunk 数
    restart_src = None        # 当前 chunk 的启动 restart (None=从 data.lammps)
    last_safe_restart = None  # 上一 chunk 的 restart (用于回滚)

    print(f"\n{'='*70}")
    print(f"  分块式主动学习")
    print(f"  每 chunk: {args.chunk_size} 步")
    print(f"  失稳阈值: score > {args.critical_threshold}")
    print(f"{'='*70}")

    while True:
        # 检查是否已达到最大轮数
        if al_iteration >= args.iters:
            print(f"\n  🏁 已完成 {al_iteration} 轮主动学习, 结束")
            break

        # 检查 MD 步数上限
        if args.max_md_steps and total_md_steps >= args.max_md_steps:
            print(f"\n  🏁 MD 总步数达到上限 {args.max_md_steps}, 结束")
            break

        # 检查 chunk 数上限
        if args.max_chunks and total_chunks >= args.max_chunks:
            print(f"\n  🏁 达到最大 chunk 数 {args.max_chunks}, 结束")
            break

        chunk_id += 1
        total_chunks += 1

        # ── ① 运行一个 MD chunk ──
        success, dump_file, restart_out, log_file = run_md_chunk(
            chunk_id, restart_src, args.chunk_size, elements=elements,
            timeout_s=args.md_timeout)
        total_md_steps += args.chunk_size

        # ── ② mlpkit.critical 判断失稳 (成功 or 崩溃都跑) ──
        trigger = "success" if success else "CRASH"
        print(f"\n    🔍 mlpkit.critical [{trigger}]: {dump_file}")
        has_critical, critical_traj = call_mlpkit_critical(
            dump_file,
            output=f'critical_{chunk_id:04d}.traj',
            score_threshold=args.critical_threshold,
            crash_score=args.critical_crash,
            baseline_frames=min(10, args.chunk_size // 100),
            min_persist=args.min_persist)

        if not has_critical:
            if success:
                # 成功且无失稳：保存当前 restart, 继续下一 chunk
                restart_path = os.path.join(META_DIR, restart_out)
                last_safe_restart = restart_path
                restart_src = restart_path
                print(f"    ✅ 无失稳, 从 {restart_out} 继续")
                continue
            else:
                # 崩溃但无失稳帧可提取 → 回滚到安全 restart 继续
                print(f"    ⚠️  Chunk {chunk_id} 崩溃但无失稳帧, 回滚继续")
                if last_safe_restart and os.path.exists(last_safe_restart):
                    restart_src = last_safe_restart
                    print(f"    🔄 回滚到 {restart_src}")
                else:
                    print(f"    ❌ 无安全 restart 可回滚, 结束")
                    break
                continue

        # ── ③ 有失稳帧 (成功检测到 or 崩溃提取到): DFT + 训练 ──
        al_iteration += 1
        reason = "mlpkit.critical 检测" if success else "MD 崩溃提取"
        print(f"\n{'='*70}")
        print(f"  🔥 主动学习轮 {al_iteration}/{args.iters}: {reason} → 失稳帧!")
        print(f"{'='*70}")

        # 复制 critical.traj 到训练目录
        samples = os.path.join(TRAIN_DIR, 'samples.traj')
        if critical_traj and os.path.exists(critical_traj):
            shutil.copy(critical_traj, samples)
            print(f"    📦 失稳帧复制: {critical_traj} → {samples}")

        # DFT
        print(f"\n  [{al_iteration}.1] DFT 计算 (siesta)...")
        dft_ok = run_dft(label=LABEL)
        if not dft_ok:
            print(f"    ❌ DFT 失败, 结束")
            break

        # 训练
        print(f"\n  [{al_iteration}.2] 训练 ReaxFF-nn ({args.epochs} epochs)...")
        run_training(args.epochs)

        # 力场同步
        print(f"\n  [{al_iteration}.3] 力场同步...")
        sync_ffield()

        # ── ④ 回滚: 从失稳 chunk 之前的安全 restart 用新力场继续 ──
        if success:
            # MD 成功完成的 chunk 检测到失稳: 回滚到上一安全 restart
            if last_safe_restart and os.path.exists(last_safe_restart):
                restart_src = last_safe_restart
                print(f"\n  🔄 回滚到安全 restart: {last_safe_restart}")
            else:
                restart_src = None
                print(f"\n  🔄 无安全 restart, 从 data.lammps 重新开始")
            # 清理本 chunk 的 restart (已污染)
            bad_restart = os.path.join(META_DIR, restart_out)
            if os.path.exists(bad_restart):
                os.remove(bad_restart)
        else:
            # MD 崩溃: restart 可能未写入或已污染, 回滚到上一安全点
            if last_safe_restart and os.path.exists(last_safe_restart):
                restart_src = last_safe_restart
                print(f"\n  🔄 (崩溃) 回滚到安全 restart: {last_safe_restart}")
            else:
                restart_src = None
                print(f"\n  🔄 (崩溃) 无安全 restart, 从 data.lammps 重新开始")
            # 清理崩溃产生的 restart
            bad_restart = os.path.join(META_DIR, restart_out)
            if os.path.exists(bad_restart):
                os.remove(bad_restart)

        print(f"     (使用新力场, 偏置势状态保持)")
        print(f"\n  ✅ 主动学习轮 {al_iteration} 完成. "
              f"用新力场继续 MD...")

        if al_iteration >= args.iters:
            print(f"\n  🏁 已完成 {al_iteration} 轮, 结束")
            break

    # ── 收尾 ──
    print(f"\n{'='*70}")
    print(f"  主动学习循环结束")
    print(f"    总 chunk 数: {total_chunks}")
    print(f"    总 MD 步数:  {total_md_steps}")
    print(f"    DFT+训练轮数: {al_iteration}/{args.iters}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

    