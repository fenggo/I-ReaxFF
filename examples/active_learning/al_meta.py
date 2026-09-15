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

# ============================================================
# 配置 (按实际环境修改)
# ============================================================
META_DIR    = '/home/feng/mlff/btf/meta'            # metaD 工作目录
TNT_DIR     = '/home/feng/mlff/btf'        # 训练工作目录
LMP         = 'lammps'
MPIRUN      = 'mpirun'
SIESTA      = 'siesta'

# EXTRACT     = os.path.join(META_DIR, 'extract_critical_frames.py')
# LMD_LM      = os.path.join(TNT_DIR, 'lm.py')
TRAIN       = os.path.join(TNT_DIR, 'train.py')
GEN         = os.path.join(TNT_DIR, 'cb22.gen')       # 共晶初始结构
FFIELD      = os.path.join(TNT_DIR, 'ffield.json')    # 训练出的力场
NPROCS      = 12

# metaD 输入 (在 META_DIR 下)
META_IN     = os.path.join(META_DIR, 'in.meta_nvt.lammps')
COLVARS     = os.path.join(META_DIR, 'colvars.meta_nvt')
DUMP        = os.path.join(META_DIR, 'meta_nvt.lammpstrj')
LOG_FILE    = os.path.join(META_DIR, 'meta_nvt.log')


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


def frame_max_force(atoms):
    if 'forces' not in atoms.arrays:
        return None
    return float(np.max(np.linalg.norm(atoms.arrays['forces'], axis=1)))


def frame_mean_force(atoms):
    if 'forces' not in atoms.arrays:
        return None
    return float(np.mean(np.linalg.norm(atoms.arrays['forces'], axis=1)))


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


def extract_unstable_frames(dump_path=None, factor=3.0, crash_factor=200.0,
                            mean_crash_factor=8.0, baseline_frames=10,
                            one_per_run=True):
    """从 dump 提取每 run 第一个失稳帧 (力异常但键完好), 写 samples.traj (新建).

    判定: 临界帧 = maxF > 3×基线 且 meanF < 8×基线 (未失稳).
          崩溃 = meanF ≥ 8×基线 或 maxF ≥ 200×基线 → 丢弃后续帧.
    返回 samples.traj 路径, 无失稳帧返回 None.
    """
    from ase.io import read, write
    if dump_path is None:
        dump_path = DUMP
    if not os.path.exists(dump_path):
        print(f"    ❌ dump 不存在: {dump_path}")
        return None

    logfile = os.path.join(META_DIR, 'lmp_meta_nvt_prod.log')
    epair_map = read_epair_from_log(logfile)

    # 第一遍: 基线 (前 N 帧)
    base_max, base_mean = [], []
    for atoms in iter_frames(dump_path, baseline_frames, epair_map):
        fm, fme = frame_max_force(atoms), frame_mean_force(atoms)
        if fm is not None:
            base_max.append(fm)
            if fme is not None:
                base_mean.append(fme)
    if not base_max:
        print("    ❌ 无法读取帧 (无 force 信息)")
        return None
    baseline = float(np.median(base_max))
    baseline_mean = float(np.median(base_mean)) if base_mean else 0.0
    threshold = baseline * factor
    crash_thr = baseline * crash_factor
    mean_crash_thr = baseline_mean * mean_crash_factor if baseline_mean > 0 else float('inf')
    print(f"    📊 基线 max|F|={baseline:.2f} mean|F|={baseline_mean:.2f} eV/Å")
    print(f"       异常阈值={threshold:.2f}, 崩溃阈值 maxF={crash_thr:.0f} meanF={mean_crash_thr:.2f}")

    # 第二遍: 取每 run 第一个失稳帧
    frames, crashed, taken = [], False, False
    stats = {"total": 0, "after_crash": 0, "anomalous": 0}
    for atoms in iter_frames(dump_path, None, epair_map):
        fm = frame_max_force(atoms)
        stats["total"] += 1
        if fm is None:
            continue
        fmean = frame_mean_force(atoms) or 0.0
        if not crashed and (fm > crash_thr or fmean > mean_crash_thr):
            crashed = True
            print(f"    💥 检测到崩溃: step {atoms.info.get('timestep')} "
                  f"(maxF={fm:.0f}, meanF={fmean:.0f}) → 丢弃后续帧")
        if crashed:
            stats["after_crash"] += 1
            continue
        step = atoms.info.get("timestep", stats["total"])
        is_anom = fm > threshold and fmean < mean_crash_thr
        if is_anom and one_per_run and taken:
            stats["after_crash"] += 1
            continue
        if is_anom:
            taken = True
            a = atoms.copy()
            a.info['maxF'] = fm
            a.info['meanF'] = fmean
            a.info['step'] = step
            a.info['source'] = os.path.basename(dump_path)
            a.info['class'] = 'anomalous'
            frames.append(a)
            stats["anomalous"] += 1
            print(f"    ⚠️  失稳帧: step {step} maxF={fm:.1f} meanF={fmean:.1f} eV/Å")

    if not frames:
        print("    ⚠️  没有提取到失稳帧!")
        return None

    samples = os.path.join(TNT_DIR, 'samples.traj')
    # 新建模式: 每轮重建 (历史样本已存 data/ct4-N.traj)
    write(samples, frames)
    print(f"    ✅ samples.traj (新建): {len(frames)} 帧失稳帧 "
          f"(扫描 {stats['total']} 帧, 丢弃崩溃后 {stats['after_crash']} 帧)")
    return samples


def register_new_data(label='ct4'):
    """把 DFT 结果注册为 data/ct4-N.traj 并加入 train.py 的 dataset dict.

    返回新数据的 key (如 'ct4-2') 或 None.
    """
    data_dir = os.path.join(TNT_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 找下一个编号: data/ct4-0, ct4-1, ... 或 dataset 里已有的最大 N
    import re
    existing = set()
    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            m = re.match(rf'{label}-(\d+)\.traj', fn)
            if m:
                existing.add(int(m.group(1)))
    # 也看 train.py dataset (跳过被注释的行)
    src = open(TRAIN).read()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        m = re.match(rf"'{label}-(\d+)'\s*:\s*'data/{label}-\d+\.traj'", stripped)
        if m:
            existing.add(int(m.group(1)))

    n = 0
    while n in existing:
        n += 1
    new_key = f"{label}-{n}"

    # DFT 输出 ct4.traj → data/ct4-N.traj
    dft_out = os.path.join(TNT_DIR, f"{label}.traj")
    if not os.path.exists(dft_out):
        print(f"    ⚠️  没有 DFT 输出 {dft_out}")
        return None
    shutil.copy(dft_out, os.path.join(data_dir, f"{new_key}.traj"))
    print(f"    📦 {dft_out} → data/{new_key}.traj")

    # 加入 train.py dataset (在 ct4-0 行后插入)
    if f"'{new_key}'" not in src:
        anchor = f"'{label}-0'  : 'data/{label}-0.traj',"
        if anchor in src:
            new_line = f"\n              '{new_key}' : 'data/{new_key}.traj',"
            src = src.replace(anchor, anchor + new_line, 1)
            with open(TRAIN, 'w') as f:
                f.write(src)
            print(f"    🔧 train.py dataset 加入 {new_key}")
    return new_key


def run_dft(label='ct4', ncpu=None):
    """siesta DFT 单点: 对 samples.traj 每帧算能量/力, 输出 <label>.traj.

    直接调用 irff 的 SinglePointEnergies (只收集 DFT 数据, 不用 lm.py 的
    主动学习循环). 需要:
      - PATH 含 siesta (~/.local/bin) 和 mpirun (mathlib) — irff 用
        `system('mpirun -n N siesta<in.fdf>siesta.out')` 起进程
      - pseudo/*.psf 拷到工作目录 — irff 的 single_point 不自动拷
    """
    if ncpu is None:
        ncpu = NPROCS
    # 必须在导入 irff 前设置 PATH
    os.environ['PATH'] = ('/home/xuni/.local/bin:'
                          '/home/xuni/siesta/mathlib/openmpi-gnu/bin:'
                          + os.environ.get('PATH', ''))

    cwd = os.getcwd()
    os.chdir(TNT_DIR)

    # pseudo 赝势 (siesta 在 in.fdf 里引用 ./C.psf 等)
    pseudo_dir = os.path.join(TNT_DIR, 'pseudo')
    if not os.path.isdir(pseudo_dir):
        print("    ❌ 需要 pseudo/ 目录 (siesta 赝势)")
        os.chdir(cwd)
        return False
    for psf in ['C.psf', 'H.psf', 'N.psf', 'O.psf']:
        src_psf = os.path.join(pseudo_dir, psf)
        if os.path.exists(src_psf) and not os.path.exists(psf):
            shutil.copy(src_psf, psf)

    traj = 'samples.traj'
    if not os.path.exists(traj):
        print(f"    ❌ 找不到 {traj}")
        os.chdir(cwd)
        return False

    from ase.io.trajectory import Trajectory
    from irff.dft.SinglePointEnergy import SinglePointEnergies

    nframe = len(Trajectory(traj))
    print(f"    🔬 siesta DFT 单点: {traj} → {label}.traj ({nframe} 帧, "
          f"xcf=GGA xca=PBE basis=split ncpu={ncpu})")

    E, E_, dEmax, d2Emax, ind_ = SinglePointEnergies(
        traj=traj,
        label=label,
        EngTole=0.01,
        frame=nframe,
        select=False,
        dE=0.1,
        colmin=5,
        dft='siesta',
        kpts=(1, 1, 1),
        xcf='GGA', xca='PBE', basistype='split',
        cpu=ncpu,
    )

    out = os.path.join(TNT_DIR, f"{label}.traj")
    if os.path.exists(out):
        from ase.io import read
        labeled = read(out, index=':')
        print(f"    ✅ DFT 完成: {len(labeled)} 帧带标签 → {label}.traj")
        os.chdir(cwd)
        return True
    print("    ⚠️ DFT 未生成输出!")
    os.chdir(cwd)
    return False


def sync_ffield():
    """把训练产物 ffield.json 转成文本 ffield 并同步到 MD 工作目录.

    train.py 输出 tnt/ffield.json (json) → mlpkit.core.ffield() → tnt/ffield (文本)
    → 拷贝到 meta/ffield (MD 的 pair_coeff 读它).
    注意: mlpkit 用本地完整版 /home/xuni/mlpkit (pip 装的 deb_bo 缺失).
    """
    # 1. json → 文本 ffield (直接调 mlpkit.core.ffield, 不另起命令)
    sys.path.insert(0, '/home/xuni/mlpkit')
    from mlpkit.core import ffield as mlpkit_ffield
    mlpkit_ffield(jsonfile=os.path.join(TNT_DIR, 'ffield.json'),
                  ffieldfile=os.path.join(TNT_DIR, 'ffield'))
    print("    🔧 ffield.json → ffield (mlpkit.core.ffield)")
    src = os.path.join(TNT_DIR, 'ffield')
    dst = os.path.join(META_DIR, 'ffield')
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"    🔄 力场同步: {src} → {dst}")
        return True
    print("    ⚠️ 没有生成 ffield, 力场未同步")
    return False


def run_training(epochs):
    """训练 ReaxFF-nn"""
    run_cmd([sys.executable, TRAIN, f'--e={epochs}'], cwd=TNT_DIR,
            timeout=8*3600, check=False)


def main():
    ap = argparse.ArgumentParser(description='主动学习循环: metaD → 提取 → DFT → 训练')
    ap.add_argument('--iters', type=int, default=1, help='迭代轮数 (默认 1)')
    ap.add_argument('--epochs', type=int, default=300, help='每轮训练 epoch (默认 300)')
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
        samples = extract_unstable_frames()
        if samples is None:
            print("    无失稳帧, 结束")
            break

        # 3. DFT
        print("\n[3/4] DFT 计算 (siesta)...")
        run_dft()

        # 3.5 注册新数据到 train.py dataset
        new_key = register_new_data('ct4')
        if new_key is None:
            print("    ⚠️ 没有新 DFT 数据, 结束")
            break

        # 4. 训练
        print("\n[4/4] 训练 ReaxFF-nn...")
        run_training(args.epochs)

        # 4.5 力场同步: ffield.json → ffield → meta/ffield (供下一轮 MD)
        sync_ffield()

        print(f"\n✅ 迭代 {it} 完成. 下一轮将用更新后的力场.")

    print("\n🏁 主动学习循环结束.")


if __name__ == '__main__':
    main()
