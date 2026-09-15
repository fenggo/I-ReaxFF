#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
al_meta.py — 主动学习(Active Learning)循环脚本
================================================
针对 TNT4·CL20 共晶 (228 atoms, ct4 体系) 的 ReaxFF-nn 力场主动学习。

流程 (每一轮迭代):
    1. MetaD MD 模拟  (in.meta_nvt_prod.lammps, NVT + coordNum CV + harmonicWalls)
         → 运行直到崩溃 (Lost atoms / hbond overflow / 数值爆炸)
    2. 提取失稳帧     (extract_critical_frames.py → samples.traj)
         → 只取每 run 第一个失稳帧 (力异常但键完好, 主动学习 hard example)
    3. DFT 计算       (lm.py, siesta 单点算能量/力, 打标签)
    4. 训练           (train.py --e=300, 更新 ffield.json)
    5. 下一轮用新力场跑 MD

用法:
    python al_meta.py                      # 运行 1 轮
    python al_meta.py --iters 5            # 运行 5 轮
    python al_meta.py --epochs 500         # 每轮训练 500 epoch
    python al_meta.py --max-md-steps 200000  # 每轮 MD 最多 20 万步 (防卡死)

依赖:
    - /home/xuni/.local/bin/lammps (ReaxFF-nn + COLVARS)
    - /home/xuni/.local/bin/siesta (DFT 单点)
    - /home/xuni/meta/tnt/ 工作目录 (train.py, lm.py, ct4.gen, ffield.json)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from ase import Atoms
from ase.io import write

# LAMMPS real → ASE 单位转换 (与 irff/lmd.py 一致, ase.calculators.lammps.unitconvert 官方因子)
from ase.calculators.lammps import unitconvert
_REAL_FORCE_TO_ASE = (unitconvert.UNITSETS['real']['force']
                      / unitconvert.UNITSETS['ASE']['force'])   # kcal/mol/Å → eV/Å
_REAL_ENERGY_TO_ASE = (unitconvert.UNITSETS['real']['energy']
                       / unitconvert.UNITSETS['ASE']['energy'])  # kcal/mol → eV
LMP         = 'lammps'
MPIRUN      = 'mpirun'
SIESTA      = 'siesta'

#=============================================================
#                  配置 (按实际环境修改)                        =
#=============================================================
META_DIR    = '/home/feng/mlff/tnt/meta'   # metaD 工作目录
TRAIN_DIR   = '/home/feng/mlff/tnt'        # 训练工作目录
LABEL       = 'ct4'
NPROCS      = 12
#=============================================================
# metaD 输入 (在 META_DIR 下)
META_IN     = os.path.join(META_DIR, 'in.meta_nvt.lammps')
COLVARS     = os.path.join(META_DIR, 'colvars.meta_nvt')
DUMP        = os.path.join(META_DIR, 'meta_nvt.lammpstrj')
LOG_FILE    = os.path.join(META_DIR, 'meta_nvt.log')
#=============================================================


TRAIN       = os.path.join(TRAIN_DIR, 'train.py')

GEN         = os.path.join(TRAIN_DIR, f'{LABEL}.gen')     # 共晶初始结构
FFIELD      = os.path.join(TRAIN_DIR, 'ffield.json')      # 训练出的力场


def run_cmd(cmd, timeout=None, cwd=None, check=True, shell=False):
    """运行命令, 捕获输出"""
    print(f"\n>>> {' '.join(cmd) if not shell else cmd}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, shell=shell,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout)
        dt = time.time() - t0
        tail = proc.stdout[-1500:] if proc.stdout else ''
        if proc.returncode != 0:
            print(f"    [exit {proc.returncode}] ({dt:.0f}s)")
            print(tail[-800:])
            if check:
                raise RuntimeError(f"命令失败: {cmd[0]} (exit {proc.returncode})")
        else:
            print(f"    [ok] ({dt:.0f}s)")
        return proc
    except subprocess.TimeoutExpired:
        print(f"    [timeout after {timeout}s]")
        return None


def md_exists_and_healthy(log):
    """检查 metaD log: 是否崩溃 (ERROR/Lost atoms/Non-numeric)"""
    if not os.path.exists(log):
        return False
    with open(log, errors='ignore') as f:
        txt = f.read()
    bad = ['ERROR', 'Lost atoms', 'Non-numeric', 'NaN', 'not enough space',
           'MPI_ABORT', 'simulation unstable']
    return not any(b in txt for b in bad)


def run_metadynamics(max_steps=1000000, timeout_s=6*3600):
    """运行 metaD MD 直到崩溃. 返回 True 若正常结束, False 若崩溃. """
    # 清旧状态 (换 CV 后必须)
    for f in ['out.colvars.state', 'out.colvars.state.old', 'out.colvars.traj',
              'out.pmf', 'rest.colvars.state']:
        p = os.path.join(META_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    # 备份旧 dump
    if os.path.exists(DUMP):
        shutil.move(DUMP, DUMP + f'.bak.{int(time.time())}')

    cmd = [MPIRUN, '-np', str(NPROCS), LMP, '-in', META_IN]
    # print(cmd)
    try:
        proc = subprocess.run(cmd, cwd=META_DIR,
                              stdout=open(LOG_FILE, 'w'), stderr=subprocess.STDOUT,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print("    ⚠️  MD 超时, 终止 (记录已有 dump)")
        return False

    if not md_exists_and_healthy(LOG_FILE):
        print("    💥 MD 崩溃 (预期, 用于提取失稳帧)")
        return False
    return True


# ============================================================
# 失稳帧提取 (原 extract_critical_frames.py 的核心逻辑, 内联)
# ============================================================

def parse_cell(bounds, tilt_labels):
    """从 LAMMPS box bounds 构建 ASE triclinic cell (含 tilt). 同 irff 公式."""
    diagdisp = np.array([bounds[0][0], bounds[0][1],
                         bounds[1][0], bounds[1][1],
                         bounds[2][0], bounds[2][1]])
    if len(bounds[0]) > 2:
        offdiag = np.array([b[2] for b in bounds])
        if len(tilt_labels) >= 3:
            order = [tilt_labels.index(t) for t in ("xy", "xz", "yz")]
            offdiag = offdiag[order]
    else:
        offdiag = np.zeros(3)
    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag
    cell = np.array([[xhi - xlo - abs(xy) - abs(xz), 0, 0],
                     [xy, yhi - ylo - abs(yz), 0],
                     [xz, yz, zhi - zlo]])
    return cell


# ── 共价键参考长度 (C-H-N-O 含能材料体系) ──
_COVALENT_BONDS = {
    ('C','C'): (1.54, 1.80), ('C','H'): (1.09, 1.20),
    ('C','N'): (1.47, 1.65), ('C','O'): (1.43, 1.55),
    ('N','N'): (1.45, 1.65), ('N','O'): (1.40, 1.60),
    ('O','O'): (1.48, 1.65), ('H','N'): (1.01, 1.15),
    ('H','O'): (0.97, 1.10), ('H','H'): (0.74, 0.90),
}


def frame_force_stats(atoms):
    """返回力的统计量: max, mean, std, pct_high (高于均值1.5倍的原子占比)."""
    if 'forces' not in atoms.arrays:
        return None
    fn = np.linalg.norm(atoms.arrays['forces'], axis=1)
    return {
        'maxF': float(np.max(fn)),
        'meanF': float(np.mean(fn)),
        'stdF': float(np.std(fn)),
        'pct_highF': float(np.mean(fn > 1.5 * np.mean(fn))) * 100,
    }


def frame_bond_stats(atoms, max_bond_dist=2.0, stretch_factor=1.15):
    """统计断键数: 对距离 < max_bond_dist 的原子对, 若超出共价键上限则计为断键.

    返回 (n_stretched, n_broken, n_close_pairs). 避免 O(N²) MIC 距离矩阵溢出.
    """
    from ase.geometry import get_distances
    natoms = len(atoms)
    D, D_len = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    stretched, broken, close = 0, 0, 0
    for i in range(natoms):
        row = D_len[i]
        for j in range(i + 1, natoms):
            d = row[j]
            if d > max_bond_dist:
                continue
            close += 1
            si, sj = sorted([atoms.symbols[i], atoms.symbols[j]])
            if (si, sj) in _COVALENT_BONDS:
                ref, max_ok = _COVALENT_BONDS[(si, sj)]
                if d > max_ok:
                    broken += 1
                elif d > ref * stretch_factor:
                    stretched += 1
    return stretched, broken, close


def frame_disp_stats(atoms, prev_atoms):
    """帧间原子位移统计 (近似速度): max displacement, RMSD."""
    if prev_atoms is None:
        return None
    disp = atoms.positions - prev_atoms.positions
    norms = np.linalg.norm(disp, axis=1)
    return {
        'max_disp': float(np.max(norms)),
        'rmsd': float(np.sqrt(np.mean(np.sum(disp**2, axis=1)))),
    }


def read_epair_from_log(logfile, units='real'):
    """从 LAMMPS log 读每个 thermo 步的 E_pair, 转 eV. 返回 {step: epair_eV}."""
    if not logfile or not os.path.exists(logfile):
        return {}
    epair = {}
    try:
        with open(logfile) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if 'Step' in lines[i] and 'E_pair' in lines[i]:
                cols = lines[i].split()
                epair_col = cols.index('E_pair')
                step_col = cols.index('Step')
                i += 1
                while i < len(lines) and not lines[i].startswith('Loop'):
                    p = lines[i].split()
                    if len(p) == len(cols):
                        try:
                            st = int(p[step_col]); ep = float(p[epair_col])
                        except (ValueError, IndexError):
                            i += 1
                            continue
                        if units == 'real':
                            ep *= _REAL_ENERGY_TO_ASE
                        epair[st] = ep
                    i += 1
                break
            i += 1
    except Exception as e:
        print(f"  ⚠️ 读能量 {logfile} 失败: {e}")
    return epair


def iter_frames(path, max_frames=None, epair_map=None):
    """迭代读取 dump 帧 (手动解析, 支持 xu/yu/zu + fx/fy/fz, triclinic cell).

    返回 ase.Atoms (forces 已转 eV/Å, info['energy'] 若提供 epair_map).
    """
    if epair_map is None:
        epair_map = {}
    elem_map = {1: 'C', 2: 'H', 3: 'N', 4: 'O'}
    n = 0
    with open(path) as f:
        lines = f.readlines()
    i, N = 0, len(lines)
    while i < N:
        if not lines[i].startswith('ITEM: TIMESTEP'):
            i += 1
            continue
        step = int(lines[i+1].strip()); i += 2
        assert lines[i].startswith('ITEM: NUMBER OF ATOMS'), lines[i]
        natoms = int(lines[i+1].strip()); i += 2
        assert lines[i].startswith('ITEM: BOX BOUNDS'), lines[i]
        tilt_labels = lines[i].split()[3:]
        bounds = []
        for _ in range(3):
            bounds.append([float(x) for x in lines[i+1].split()]); i += 1
        i += 1  # blank line
        cols = lines[i].split()[2:]; i += 1
        def colidx(name):
            return cols.index(name) if name in cols else None
        ix = colidx('xu') or colidx('x')
        iy = colidx('yu') or colidx('y')
        iz = colidx('zu') or colidx('z')
        ifx, ify, ifz = colidx('fx'), colidx('fy'), colidx('fz')
        itype = colidx('type')

        symbols, pos, forces = [], [], None
        for _ in range(natoms):
            p = lines[i].split(); i += 1
            symbols.append(elem_map.get(int(p[itype]), 'X') if itype is not None else 'X')
            pos.append([float(p[ix]), float(p[iy]), float(p[iz])])
            if ifx is not None:
                if forces is None:
                    forces = []
                forces.append([float(p[ifx]), float(p[ify]), float(p[ifz])])
        pos = np.array(pos)
        cell = parse_cell(bounds, tilt_labels)
        atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=[True]*3)
        if forces is not None:
            atoms.set_array('forces', np.array(forces) * _REAL_FORCE_TO_ASE)
        atoms.info['timestep'] = step
        if step in epair_map:
            atoms.info['energy'] = epair_map[step]
        yield atoms
        n += 1
        if max_frames is not None and n >= max_frames:
            break


def _compute_stability_score(fstats, bstats, dstats, baselines):
    """综合稳定性评分 (0=正常, 越高越不稳定). 多信号加权:

    信号                    检测什么              沉没成本敏感度
    ─────────────────────────────────────────────────────────────
    力 Z-score (maxF)       局部力异常开始         中等 — 力飙时结构已坏
    highF% 原子占比         力分布变宽             高 — 比 maxF 早 1-2 帧
    断键数增量             键开始断裂             最高 — 最早的失稳信号
    max_disp                原子开始位移            高 — 力学不稳定开始
    RMSD                    全局结构漂移            低 — 滞后, 作为确认
    """
    score = 0.0
    flags = []

    # 1. maxF Z-score: 力偏离基线几个标准差
    fz = (fstats['maxF'] - baselines['maxF_mean']) / max(baselines['maxF_std'], 1e-6)
    if fz > 2.0:
        score += min(fz - 2.0, 8.0)  # cap at +8
        flags.append(f'F{fz:.1f}')

    # 2. highF% 增量 (力分布尾部变胖是更早的信号)
    hp_delta = fstats['pct_highF'] - baselines['highF_mean']
    if hp_delta > baselines['highF_std'] * 2:
        score += min(hp_delta / 5, 6.0)
        flags.append(f'hF+{hp_delta:.0f}%')

    # 3. 断键数 — 最重要的信号, 权重最高
    bdelta = bstats[1] - baselines['broken_mean']  # bstats = (stretched, broken, close)
    if bdelta > 0:
        score += bdelta * 2.0  # 每多一个断键 +2
        flags.append(f'Br+{bdelta}')

    # 4. 最大位移 — 原子突然开始移动
    if dstats is not None and baselines.get('max_disp_mean', 0) > 0:
        dz = (dstats['max_disp'] - baselines['max_disp_mean']) / max(baselines['max_disp_std'], 1e-8)
        if dz > 3.0:
            score += min(dz - 3.0, 5.0)
            flags.append(f'D{dz:.1f}')

    # 5. RMSD — 全局漂移确认
    if dstats is not None and baselines.get('rmsd_mean', 0) > 0:
        rz = (dstats['rmsd'] - baselines['rmsd_mean']) / max(baselines['rmsd_std'], 1e-8)
        if rz > 5.0:
            score += min(rz - 5.0, 3.0)
            flags.append(f'R{rz:.1f}')

    return score, flags


def extract_critical_frames(dump_path=None, score_threshold=4.0,
                            crash_score=30.0, baseline_frames=10,
                            one_per_run=True):
    """多信号综合评判提取失稳帧, 写 samples.traj (新建).

    不依赖单一阈值, 而是对每个帧计算综合稳定性评分:
      - 力分布 (maxF Z-score, 高力原子占比)
      - 键完整性 (超出共价键上限的断键数)
      - 原子位移 (帧间 max displacement, RMSD)

    评分 > score_threshold 标志着"失稳开始", 取其**前一帧** (结构仍完好).
    评分 > crash_score 标志着"已崩溃", 丢弃该帧及后续.

    返回 samples.traj 路径, 无失稳帧返回 None.
    """
    from ase.io import read, write
    from ase.calculators.singlepoint import SinglePointCalculator
    if dump_path is None:
        dump_path = DUMP
    if not os.path.exists(dump_path):
        print(f"    ❌ dump 不存在: {dump_path}")
        return None

    logfile = os.path.join(META_DIR, 'lmp_meta_nvt_prod.log')
    epair_map = read_epair_from_log(logfile)

    # ── 第一遍: 收集基线 (前 baseline_frames 帧) ──
    base_fstats = []  # dicts from frame_force_stats
    base_bstats = []  # (stretched, broken, close)
    base_dstats = []  # dicts from frame_disp_stats
    prev = None
    for atoms in iter_frames(dump_path, baseline_frames, epair_map):
        fs = frame_force_stats(atoms)
        if fs is None:
            continue
        base_fstats.append(fs)
        base_bstats.append(frame_bond_stats(atoms))
        ds = frame_disp_stats(atoms, prev)
        if ds is not None:
            base_dstats.append(ds)
        prev = atoms

    if not base_fstats:
        print("    ❌ 无法读取帧 (无 force 信息)")
        return None

    baselines = {
        'maxF_mean': float(np.mean([f['maxF'] for f in base_fstats])),
        'maxF_std':  float(np.std([f['maxF'] for f in base_fstats])),
        'highF_mean': float(np.mean([f['pct_highF'] for f in base_fstats])),
        'highF_std':  float(np.std([f['pct_highF'] for f in base_fstats])),
        'broken_mean': float(np.mean([b[1] for b in base_bstats])),
    }
    if base_dstats:
        baselines['max_disp_mean'] = float(np.mean([d['max_disp'] for d in base_dstats]))
        baselines['max_disp_std']  = float(np.std([d['max_disp'] for d in base_dstats]))
        baselines['rmsd_mean'] = float(np.mean([d['rmsd'] for d in base_dstats]))
        baselines['rmsd_std']  = float(np.std([d['rmsd'] for d in base_dstats]))

    print(f"    📊 基线 ({len(base_fstats)} 帧): "
          f"maxF={baselines['maxF_mean']:.1f}±{baselines['maxF_std']:.1f}, "
          f"断键均={baselines['broken_mean']:.0f}")
    print(f"       异常阈值(评分)={score_threshold}, 崩溃阈值(评分)={crash_score}")

    # ── 第二遍: 扫描所有帧, 综合评分 ──
    frames, crashed, taken = [], False, False
    prev_atoms = None
    stats = {"total": 0, "after_crash": 0, "anomalous": 0}
    score_history = []  # (step, score)

    for atoms in iter_frames(dump_path, None, epair_map):
        stats["total"] += 1

        fs = frame_force_stats(atoms)
        if fs is None:
            continue

        bs = frame_bond_stats(atoms)
        ds = frame_disp_stats(atoms, prev_atoms)  # 位移仍需前一帧

        score, flags = _compute_stability_score(fs, bs, ds, baselines)
        step = atoms.info.get("timestep", stats["total"])
        score_history.append((step, score, flags))

        if not crashed and score > crash_score:
            crashed = True
            print(f"    💥 崩溃: step {step} score={score:.1f} [{', '.join(flags)}]")
        if crashed:
            stats["after_crash"] += 1
            prev_atoms = atoms
            continue

        # 判定异常帧: 评分超过阈值, 直接提取当前帧
        is_anom = score > score_threshold
        if is_anom and one_per_run and taken:
            stats["after_crash"] += 1
            prev_atoms = atoms
            continue

        if is_anom:
            taken = True
            a = atoms.copy()
            a.info['maxF'] = fs['maxF']
            a.info['meanF'] = fs['meanF']
            a.info['broken_bonds'] = bs[1]
            a.info['step'] = step
            a.info['source'] = os.path.basename(dump_path)
            a.info['class'] = 'anomalous'
            a.info['score'] = score
            a.calc = SinglePointCalculator(
                a,
                energy=a.info.get('energy', 0.0),
                forces=a.arrays.get('forces', None),
            )
            frames.append(a)
            stats["anomalous"] += 1
            print(f"    ⚠️  失稳帧: step {step} "
                  f"maxF={fs['maxF']:.1f} broken={bs[1]} "
                  f"score={score:.1f} [{', '.join(flags)}]")

        prev_atoms = atoms

    # 打印评分历史 (前20帧 + 最后5帧, 若太长则截断)
    if score_history:
        print(f"    📈 评分历史 (frame, score):")
        n_show = min(20, len(score_history))
        for st, sc, fl in score_history[:n_show]:
            marker = " ←异常" if sc > score_threshold else ""
            print(f"       step {st:6d}  score={sc:5.1f}  [{', '.join(fl) if fl else 'ok'}]{marker}")
        if len(score_history) > n_show + 5:
            print(f"       ... ({len(score_history) - n_show - 5} 帧省略) ...")
        for st, sc, fl in score_history[-5:]:
            marker = " ←异常" if sc > score_threshold else ""
            print(f"       step {st:6d}  score={sc:5.1f}  [{', '.join(fl) if fl else 'ok'}]{marker}")

    if not frames:
        print("    ⚠️  没有提取到失稳帧! 评分历史见上.")
        return None

    samples = os.path.join(TRAIN_DIR, 'samples.traj')
    write(samples, frames)
    print(f"\n    ✅ samples.traj (新建): {len(frames)} 帧 "
          f"(扫描 {stats['total']} 帧, 丢弃崩溃后 {stats['after_crash']} 帧)")
    return samples


def run_dft(label='cb22', ncpu=None):
    """siesta DFT 单点: 运行 lm.py 对 samples.traj 每帧算能量/力, 输出 <label>.traj."""
    if ncpu is None:
        ncpu = NPROCS

    lm_script = os.path.join(TRAIN_DIR, 'lm.py')
    if not os.path.exists(lm_script):
        print(f"    ❌ 找不到 {lm_script}")
        return False

    cwd = os.getcwd()
    os.chdir(TRAIN_DIR)

    log_path = os.path.join(TRAIN_DIR, 'lm.log')
    with open(log_path, 'w') as log_fh:
        proc = subprocess.run(
            [sys.executable, lm_script],
            stdout=log_fh, stderr=subprocess.STDOUT,
            timeout=90000
        )

    os.chdir(cwd)

    out = os.path.join(TRAIN_DIR, f"{label}.traj")
    if proc.returncode != 0:
        print(f"    ❌ lm.py 失败 (exit {proc.returncode})")
        return False
    if not os.path.exists(out):
        print(f"    ⚠️ DFT 未生成 {label}.traj")
        return False

    from ase.io import read
    labeled = read(out, index=':')
    print(f"    ✅ DFT 完成: {len(labeled)} 帧带标签 → {label}.traj")
    return True


def sync_ffield():
    """把训练产物 ffield.json 转成文本 ffield 并同步到 MD 工作目录.

    train.py 输出 tnt/ffield.json (json) → mlpkit.core.ffield() → tnt/ffield (文本)
    → 拷贝到 meta/ffield (MD 的 pair_coeff 读它).
    注意: mlpkit 用本地完整版 /home/xuni/mlpkit (pip 装的 deb_bo 缺失).
    """
    # 1. json → 文本 ffield (直接调 mlpkit.core.ffield, 不另起命令)
    sys.path.insert(0, '/home/xuni/mlpkit')
    from mlpkit.core import ffield as mlpkit_ffield
    mlpkit_ffield(jsonfile=os.path.join(TRAIN_DIR, 'ffield.json'),
                  ffieldfile=os.path.join(TRAIN_DIR, 'ffield'))
    print("    🔧 ffield.json → ffield (mlpkit.core.ffield)")
    src = os.path.join(TRAIN_DIR, 'ffield')
    dst = os.path.join(META_DIR, 'ffield')
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"    🔄 力场同步: {src} → {dst}")
        return True
    print("    ⚠️ 没有生成 ffield, 力场未同步")
    return False


def run_training(epochs):
    """训练 ReaxFF-nn"""
    run_cmd([sys.executable, TRAIN, f'--e={epochs}'], cwd=TRAIN_DIR,
            timeout=8*3600, check=False)


def main():
    ap = argparse.ArgumentParser(description='主动学习循环: metaD → 提取 → DFT → 训练')
    ap.add_argument('--iters', type=int, default=1, help='迭代轮数 (默认 1)')
    ap.add_argument('--epochs', type=int, default=1000, help='每轮训练 epoch (默认 300)')
    ap.add_argument('--max-md-steps', type=int, default=100000,
                    help='每轮 MD 最大步数 (默认 1e6)')
    ap.add_argument('--md-timeout', type=int, default=6*3600,
                    help='每轮 MD 超时秒数 (默认 6h)')
    args = ap.parse_args()

    for it in range(1, args.iters + 1):
        print(f"\n{'='*60}")
        print(f"  主动学习迭代 {it}/{args.iters}")
        print(f"{'='*60}")

        # 1. metaD MD
        print("\n[1/4] metaD 模拟...")
        finished = run_metadynamics(timeout_s=args.md_timeout)
        if finished:
            print("    MD 正常结束 (未崩溃, 无失稳帧) — 提前结束")
            break

        # 2. 提取失稳帧
        print("\n[2/4] 提取失稳帧...")
        samples = extract_critical_frames(score_threshold=3.0)
        if samples is None:
            print("    无失稳帧, 结束")
            break

        # 3. DFT
        print("\n[3/4] DFT 计算 (siesta)...")
        run_dft(label=LABEL)

        # 3.5 注册新数据到 train.py dataset
        # new_key = register_new_data(LABEL)
        # if new_key is None:
        #     print("    ⚠️ 没有新 DFT 数据, 结束")
        #     break

        # 4. 训练
        print("\n[4/4] 训练 ReaxFF-nn...")
        run_training(args.epochs)

        # 4.5 力场同步: ffield.json → ffield → meta/ffield (供下一轮 MD)
        sync_ffield()

        print(f"\n✅ 迭代 {it} 完成. 下一轮将用更新后的力场.")

    print("\n🏁 主动学习循环结束.")


if __name__ == '__main__':
    main()
