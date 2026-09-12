#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_critical_frames.py — 从 LAMMPS dump 轨迹中自动提取"跑崩前临界结构"

用途:
    扫描 LAMMPS dump 文件 (*.lammpstrj),按原子受力判断异常度,
    自动保存数值不稳定边缘的结构为 ASE extended XYZ (*.extxyz),
    供 ReaxFF-nn / NequIP / MACE / DeePMD 等力场训练使用。

异常判定:
    对每个帧计算 max|F|(最大原子受力)和 mean|F|。
    基线 = 轨迹前若干帧(正常晶体)的 max|F| 中位数 × 系数(默认 3)。
    帧的 max|F| > 基线 → 标记为 anomalous(临界/异常结构)。
    同时保存 N 个正常帧作为正例。

用法:
    python extract_critical_frames.py --dumps meta_nvt.lammpstrj meta_nvt_prod.lammpstrj
    python extract_critical_frames.py --dumps *.lammpstrj --outdir critical_frames \
        --factor 3 --norm 30 --all-frames
    python extract_critical_frames.py --dumps meta_nvt.lammpstrj --extxyz tn_cl20.extxyz

输出:
    <outdir>/anomalous/       # 异常帧, 文件名带 step 和 maxF
    <outdir>/normal/          # 正常帧
    <outdir>/combined.extxyz  # 全部异常帧 + 正常帧合并(带 info: maxF, source, step)
    <outdir>/manifest.csv     # 帧清单: step, maxF, meanF, class

依赖:
    pip install ase numpy
"""

import argparse
import csv
import os
import sys

import numpy as np
from ase import Atoms
from ase.io import write

# LAMMPS real → ASE 单位转换 (与 irff/lmd.py 的 lammpstraj_to_ase 完全一致:
# 用 ase.calculators.lammps.unitconvert 的官方因子)
from ase.calculators.lammps import unitconvert
_REAL_FORCE_TO_ASE = (unitconvert.UNITSETS['real']['force']
                      / unitconvert.UNITSETS['ASE']['force'])   # kcal/mol/Å → eV/Å
_REAL_ENERGY_TO_ASE = (unitconvert.UNITSETS['real']['energy']
                       / unitconvert.UNITSETS['ASE']['energy'])  # kcal/mol → eV
# 注: 旧 lmd.py 里硬编码 energy 因子 4.3364432032e-2 与此略有差异 (~7.6e-6 相对);
# 但 lmd.py 的 traj 实际走 convert() (本因子), 用本因子与 lmd.py 输出一致.


# ============================================================
# 工具函数
# ============================================================

def list_dump_files(patterns):
    """展开通配符, 收集所有 lammpstrj 文件"""
    import glob
    files = []
    for p in patterns:
        if os.path.isfile(p):
            files.append(p)
        else:
            files.extend(glob.glob(p))
    # 去重保序
    seen = set()
    return [f for f in files if not (f in seen or seen.add(f))]


def parse_cell(bounds, tilt_labels, tilt_values):
    """从 LAMMPS box bounds 构建 ASE triclinic cell (含 tilt).

    与 irff.md.lammps.construct_cell 相同的公式:
      bounds: [[xlo,xhi],[ylo,yhi],[zlo,zhi]] (含可能的第3列 tilt)
      tilt_labels: 'ITEM: BOX BOUNDS' 行尾部的标签 (xy xz yz ...)
    """
    diagdisp = np.array([bounds[0][0], bounds[0][1],
                         bounds[1][0], bounds[1][1],
                         bounds[2][0], bounds[2][1]])

    # tilt (第3列): 按标签顺序排列为 xy, xz, yz
    if len(bounds[0]) > 2:
        offdiag = np.array([b[2] for b in bounds])
        if len(tilt_labels) >= 3:
            order = [tilt_labels.index(t) for t in ("xy", "xz", "yz")]
            offdiag = offdiag[order]
    else:
        offdiag = np.zeros(3)

    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag
    # ASE cell (LAMMPS triclinic 约定, 同 irff)
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = zhi - zlo
    cell = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    return cell


def frame_max_force(atoms):
    """帧内最大原子受力 (eV/Å 或 kcal/mol/Å, 单位随源文件)"""
    if 'forces' not in atoms.arrays:
        return None
    f = atoms.arrays['forces']
    return float(np.max(np.linalg.norm(f, axis=1)))


def frame_mean_force(atoms):
    if 'forces' not in atoms.arrays:
        return None
    f = atoms.arrays['forces']
    return float(np.mean(np.linalg.norm(f, axis=1)))


def read_epair_from_log(logfile, units='real'):
    """从 LAMMPS log 读每个 thermo 步的 E_pair, 转为 eV (与 lmd.py 一致).

    返回 {step: epair_eV}. 只解析 thermo_style 含 'E_pair' 的输出段.
    """
    if not logfile or not os.path.exists(logfile):
        return {}
    epair = {}
    try:
        with open(logfile) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            ln = lines[i]
            if 'Step' in ln and 'E_pair' in ln:
                cols = ln.split()
                epair_col = cols.index('E_pair') if 'E_pair' in cols else None
                step_col = cols.index('Step') if 'Step' in cols else None
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
                            ep = ep * _REAL_ENERGY_TO_ASE  # kcal/mol → eV
                        epair[st] = ep
                    i += 1
                break
            i += 1
    except Exception as e:
        print(f"  ⚠️ 读能量 {logfile} 失败: {e}")
    return epair


def iter_frames(path, max_frames=None, epair_map=None):
    """迭代读取 dump 帧 (惰性, 避免大文件全载入).

    手动解析 lammpstrj: 支持 id type xu/yu/zu fx/fy/fz 及任意列名,
    正确提取 positions (unwrapped), forces 与 triclinic cell.
    返回 ase.Atoms 对象 (info['energy'] 为 E_pair/eV, 若提供 epair_map).
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
        i += 1  # skip blank line
        cols = lines[i].split()[2:]; i += 1
        # 列索引
        def colidx(name):
            return cols.index(name) if name in cols else None
        ix, iy, iz = colidx('xu'), colidx('yu'), colidx('zu')
        if ix is None: ix = colidx('x')
        if iy is None: iy = colidx('y')
        if iz is None: iz = colidx('z')
        ifx, ify, ifz = colidx('fx'), colidx('fy'), colidx('fz')
        itype = colidx('type') if 'type' in cols else None

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
        cell = parse_cell(bounds, tilt_labels, [])
        atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=[True]*3)
        if forces is not None:
            # LAMMPS real (kcal/mol/Å) → ASE (eV/Å), 与 lmd.py 一致
            atoms.set_array('forces', np.array(forces) * _REAL_FORCE_TO_ASE)
        atoms.info['timestep'] = step
        if step in epair_map:
            atoms.info['energy'] = epair_map[step]
        yield atoms
        n += 1
        if max_frames is not None and n >= max_frames:
            break


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="从 LAMMPS dump 提取异常(高受力)结构为 ASE extxyz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="LAMMPS dump 文件或通配符, 如 '*.lammpstrj'")
    ap.add_argument("--outdir", default="critical_frames",
                    help="输出目录 (默认: critical_frames)")
    ap.add_argument("--factor", type=float, default=3.0,
                    help="异常阈值 = 基线maxF × factor (默认 3.0)")
    ap.add_argument("--norm", type=int, default=0,
                    help="保存的正常帧数量 (默认 0 = 不保存, 只输出异常/失稳帧; "
                         "需要正例时设 >0)")
    ap.add_argument("--baseline-frames", type=int, default=10,
                    help="用前 N 帧估基线 (默认 10)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="每个 dump 最多读多少帧 (调试用)")
    ap.add_argument("--extxyz", default=None,
                    help="合并输出文件名 (默认 <outdir>/combined.extxyz)")
    ap.add_argument("--min-maxf", type=float, default=0.0,
                    help="只保存 maxF 超过此绝对值的异常帧 (可选过滤)")
    ap.add_argument("--crash-threshold", type=float, default=200.0,
                    help="崩溃判定 maxF = 基线maxF × 此值 (默认 200.0). 只抓极端"
                         "飞散 (单原子力爆表); 主要崩溃判据用 meanF")
    ap.add_argument("--mean-crash-threshold", type=float, default=8.0,
                    help="崩溃判定 meanF = 基线meanF × 此值 (默认 8.0). meanF 超"
                         "过视为整体失稳(键断裂/飞散), 丢弃. 临界帧 = maxF 异常但 "
                         "meanF 未失稳 (刚有力异常, 化学键完好)")
    ap.add_argument("--one-per-run", action="store_true", default=True,
                    help="每个 run (dump 文件) 只取第一个失稳帧, 后续全丢 "
                         "(默认开启, 主动学习: 一个 run 一个 hard example)")
    ap.add_argument("--log", default=None,
                    help="LAMMPS log 文件 (可选, 读 E_pair 能量写入 extxyz; "
                         "支持多个, 逗号分隔, 与 --dumps 对应)")
    args = ap.parse_args()

    dumps = list_dump_files(args.dumps)
    if not dumps:
        print(f"❌ 没有找到 dump 文件: {args.dumps}")
        sys.exit(1)

    # 能量映射: 每个 dump 找对应 log (自动匹配 basename 或显式 --log)
    epair_maps = {}
    if args.log:
        logs = [l.strip() for l in args.log.split(',') if l.strip()]
        for d in dumps:
            base = os.path.basename(d).replace('.lammpstrj', '')
            cand = None
            for l in logs:
                lb = os.path.basename(l)
                # 精确匹配: lmp_<base>.log, <base>.log, 或 base 完全在 log 名中
                if lb in (f"lmp_{base}.log", f"{base}.log") or \
                   lb.startswith(base + '.') or lb.startswith('lmp_' + base + '.'):
                    cand = l
                    break
            if cand is None and len(logs) == 1:
                cand = logs[0]
            if cand:
                epair_maps[d] = read_epair_from_log(cand)
                print(f"  📄 能量来源: {cand} ({len(epair_maps[d])} 步)")
            else:
                epair_maps[d] = {}
    else:
        for d in dumps:
            epair_maps[d] = {}

    os.makedirs(os.path.join(args.outdir, "anomalous"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "normal"), exist_ok=True)

    print(f"📁 扫描 {len(dumps)} 个 dump 文件:")
    for d in dumps:
        print(f"   {d}")

    # ---- 第一遍: 估基线 (用各 dump 前几帧的正常结构) ----
    baseline_maxf_list = []
    baseline_meanf_list = []
    for d in dumps:
        try:
            for i, atoms in enumerate(iter_frames(d, args.baseline_frames, epair_maps[d])):
                fm = frame_max_force(atoms)
                if fm is not None:
                    baseline_maxf_list.append(fm)
                    fme = frame_mean_force(atoms)
                    if fme is not None:
                        baseline_meanf_list.append(fme)
        except Exception as e:
            print(f"  ⚠️ 读 {d} 失败: {e}")
    if not baseline_maxf_list:
        print("❌ 无法读取任何帧(没有 force 信息?)")
        sys.exit(1)
    baseline = float(np.median(baseline_maxf_list))
    baseline_mean = float(np.median(baseline_meanf_list)) if baseline_meanf_list else 0.0
    threshold = baseline * args.factor
    crash_thr = baseline * args.crash_threshold
    mean_crash_thr = baseline_mean * args.mean_crash_threshold if baseline_mean > 0 else float('inf')
    print(f"\n📊 基线 max|F|(前{args.baseline_frames}帧中位数) = {baseline:.2f}")
    print(f"   基线 mean|F| = {baseline_mean:.2f}")
    print(f"   异常阈值(maxF) = {baseline:.2f} × {args.factor} = {threshold:.2f}")
    print(f"   崩溃阈值(maxF) = {baseline:.2f} × {args.crash_threshold} = {crash_thr:.2f}")
    print(f"   崩溃阈值(meanF) = {baseline_mean:.2f} × {args.mean_crash_threshold} = {mean_crash_thr:.2f}")

    # ---- 第二遍: 分类并保存 ----
    manifest = []
    anomalous_frames = []
    normal_frames = []
    stats = {"total": 0, "anomalous": 0, "normal": 0, "no_force": 0,
             "crashed_at": None, "after_crash": 0}

    for d in dumps:
        print(f"\n🔍 处理 {d} ...")
        crashed = False
        taken_first = False  # 每个 run 只取第一个失稳帧
        for atoms in iter_frames(d, args.max_frames, epair_maps[d]):
            fm = frame_max_force(atoms)
            stats["total"] += 1
            if fm is None:
                stats["no_force"] += 1
                continue
            fmean = frame_mean_force(atoms) or 0.0

            # 崩溃检测: maxF 或 meanF 超过硬阈值 → 整体失稳/飞散
            if not crashed and (fm > crash_thr or fmean > mean_crash_thr):
                crashed = True
                stats["crashed_at"] = int(atoms.info.get("timestep", stats["total"]))
                print(f"    💥 检测到崩溃/失稳: step {stats['crashed_at']} "
                      f"maxF={fm:.0f} (阈 {crash_thr:.0f}), meanF={fmean:.0f} "
                      f"(阈 {mean_crash_thr:.0f}) → 丢弃后续帧")
            if crashed:
                stats["after_crash"] += 1
                continue  # 崩溃后的帧无训练价值, 丢弃

            step = int(atoms.info.get("timestep", stats["total"]))
            # 临界帧 = maxF 异常 (刚有力异常) 且 meanF 未失稳 (化学键完好)
            is_anom = (fm > threshold and (args.min_maxf == 0 or fm >= args.min_maxf)
                       and fmean < mean_crash_thr)
            cls = "anomalous" if is_anom else "normal"

            # 每个 run 只取第一个失稳帧, 后续帧全部丢弃 (主动学习: 一个 run 一个样本)
            if is_anom and args.one_per_run and taken_first:
                stats["after_crash"] += 1
                continue
            if is_anom:
                taken_first = True

            # 深拷贝 atoms 并写入 info(避免共享引用)
            a = atoms.copy()
            a.info["maxF"] = fm
            a.info["meanF"] = fmean
            a.info["source"] = os.path.basename(d)
            a.info["step"] = step
            a.info["class"] = cls

            fname = f"{d}_step{step}.extxyz" if len(dumps) > 1 else f"step{step}.extxyz"
            if is_anom:
                anomalous_frames.append(a)
                out_path = os.path.join(args.outdir, "anomalous", fname)
                write(out_path, a, format="extxyz")
                stats["anomalous"] += 1
                if stats["anomalous"] <= 10:
                    print(f"    ⚠️  step {step}: maxF={fm:.1f} → {fname}")
            elif len(normal_frames) < args.norm:
                normal_frames.append(a)
                out_path = os.path.join(args.outdir, "normal", fname)
                write(out_path, a, format="extxyz")
                stats["normal"] += 1

            manifest.append((step, fm, fmean, cls, os.path.basename(d)))

    # ---- 清单 ----
    with open(os.path.join(args.outdir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "maxF", "meanF", "class", "source"])
        w.writerows(manifest)

    # ---- 合并输出 ----
    combined = anomalous_frames + normal_frames
    if combined and args.extxyz is None:
        args.extxyz = os.path.join(args.outdir, "combined.extxyz")
    if combined:
        write(args.extxyz, combined, format="extxyz")
        print(f"\n💾 合并输出: {args.extxyz} ({len(combined)} 帧)")

    # ---- 汇总 ----
    print(f"\n{'='*50}")
    print(f"  ✅ 完成")
    print(f"  总帧数:      {stats['total']}")
    print(f"  异常帧:      {stats['anomalous']}")
    print(f"  正常帧:      {stats['normal']}")
    print(f"  无力帧(跳过): {stats['no_force']}")
    if stats['crashed_at'] is not None:
        print(f"  💥 检测到崩溃: step {stats['crashed_at']} "
              f"(丢弃崩溃后 {stats['after_crash']} 帧)")
    print(f"  输出目录:    {args.outdir}/")
    print(f"    anomalous/   ({stats['anomalous']} 个文件)")
    print(f"    normal/      ({stats['normal']} 个文件)")
    print(f"    combined.extxyz")
    print(f"    manifest.csv")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
