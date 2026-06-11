#!/usr/bin/env python
"""
ase_to_gap.py — 将 ASE trajectory (.traj) 转换为 GAP (gap_fit) 训练用的 extended XYZ 格式。

用法：
    cd 到包含 .traj 文件的目录，然后：
    python ase_to_gap.py                        # 转换当前目录所有 .traj
    python ase_to_gap.py --t md                 # 只转换文件名以 "md" 开头的 .traj
    python ase_to_gap.py --t "md aimd"          # 多个前缀，空格分隔
    python ase_to_gap.py --o train.xyz          # 指定输出文件名
    python ase_to_gap.py --t md --o gap_data.xyz

输出格式：GAP extended XYZ
  - 每帧第一行：原子数
  - 第二行：Lattice, Properties, energy, force, virial, config_type, pbc
  - 后续行：species pos_x pos_y pos_z fx fy fz
  - 单位：能量 eV，力 eV/Å，应力 eV (stress × volume)

注意事项：
  - VASP 用户：VASP 应力符号与 QUIP 相反，可能需要手动翻转 virial 符号
  - 即使是分子/非周期体系，也必须提供 Lattice（用大于 2×cutoff 的大 box）
  - 如果数据中能量已是结合能，需要在数据中或命令行指定 e0=0
"""

import sys
import argparse
import os
import numpy as np
from ase.io.trajectory import Trajectory


def write_extxyz(f, atoms, frame_idx, config_type="unknown"):
    """
    将单个 ASE Atoms 对象写入 extended XYZ 格式。

    Parameters
    ----------
    f : file object
    atoms : ase.Atoms
    frame_idx : int
    config_type : str
        结构类型标签，用于 gap_fit 的 config_type_sigma 分组正则化。
    """
    natom = len(atoms)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()       # (N, 3), Å
    cell = atoms.get_cell()                 # (3, 3), Å
    pbc = atoms.get_pbc()                   # (3,) bool

    # --- 能量 ---
    energy = atoms.get_potential_energy()   # eV

    # --- 力 ---
    forces = atoms.get_forces()             # (N, 3), eV/Å

    # --- 维里应力 (可选) ---
    # gap_fit 需要 virial = stress × volume，单位 eV，9分量 (xx, xy, xz, yx, yy, yz, zx, zy, zz)
    has_virial = False
    virial_flat = None
    try:
        stress = atoms.get_stress(voigt=False)  # (3,3) 或 (6,) 取决于 ASE 版本
        if stress is not None:
            vol = atoms.get_volume()
            if stress.shape == (6,):
                # Voigt 顺序: xx, yy, zz, yz, xz, xy → 转为 3x3
                s = stress
                virial_3x3 = np.array([
                    [s[0], s[5], s[4]],
                    [s[5], s[1], s[3]],
                    [s[4], s[3], s[2]]
                ]) * vol
            elif stress.shape == (3, 3):
                virial_3x3 = stress * vol
            else:
                virial_3x3 = None

            if virial_3x3 is not None:
                virial_flat = virial_3x3.flatten()  # 9 分量
                has_virial = True
    except Exception:
        pass

    # --- 构建 Properties 字符串 ---
    # 最少: species:S:1:pos:R:3
    prop_parts = ["species:S:1", "pos:R:3"]

    # 如果有力，加入 force:R:3
    has_force = forces is not None
    if has_force:
        prop_parts.append("force:R:3")

    properties = "Properties=" + ":".join(prop_parts)

    # --- 构建第二行 (comment line) ---
    # Lattice: 9 个浮点数 (3 个晶格矢量展平)
    lat_flat = cell.flatten()
    lattice_str = " ".join(f"{v:15.8f}" for v in lat_flat)
    lattice_kv = f'Lattice="{lattice_str}"'

    # pbc
    pbc_str = " ".join("T" if p else "F" for p in pbc)
    pbc_kv = f'pbc="{pbc_str}"'

    # config_type
    ct_kv = f"config_type={config_type}"

    # 构建第二行
    comment_parts = [ct_kv, lattice_kv, pbc_kv, properties]

    if energy is not None:
        comment_parts.insert(1, f"energy={energy:15.8f}")

    if has_virial and virial_flat is not None:
        virial_str = " ".join(f"{v:15.8f}" for v in virial_flat)
        comment_parts.insert(1, f'virial="{virial_str}"')

    comment_line = " ".join(comment_parts)

    # --- 写入 ---
    f.write(f"{natom}\n")
    f.write(f"{comment_line}\n")

    for i in range(natom):
        # 原子行: species x y z [fx fy fz]
        parts = [f"{symbols[i]:3s}"]
        parts.append(f"{positions[i][0]:15.8f} {positions[i][1]:15.8f} {positions[i][2]:15.8f}")
        if has_force:
            parts.append(f"{forces[i][0]:15.8f} {forces[i][1]:15.8f} {forces[i][2]:15.8f}")
        f.write(" ".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="将 ASE .traj 文件转换为 GAP gap_fit 训练用的 extended XYZ 格式"
    )
    parser.add_argument(
        "--t", default="",
        type=str,
        help="trajectory 文件名前缀过滤，多个前缀用空格分隔并用引号包裹，如 --t 'md aimd'"
    )
    parser.add_argument(
        "--o", default="train.xyz",
        type=str,
        help="输出文件名 (默认: train.xyz)"
    )
    parser.add_argument(
        "--config-type", default="auto",
        type=str,
        help="config_type 来源: auto(从文件名提取), traj(用traj文件名), 或直接指定字符串"
    )
    parser.add_argument(
        "--skip-no-force", action="store_true",
        help="跳过没有力数据的帧 (默认: 包含但 force 列不写入)"
    )
    parser.add_argument(
        "--no-force", action="store_true",
        help="不写入力数据 (即使 traj 中有)"
    )
    args = parser.parse_args(sys.argv[1:])

    cdir = os.getcwd()
    files = os.listdir(cdir)

    # --- 筛选 .traj 文件 ---
    if args.t:
        prefixes = args.t.split()
        trajs = []
        for prefix in prefixes:
            for fil in files:
                if fil.startswith(prefix) and fil.endswith(".traj"):
                    if fil not in trajs:
                        trajs.append(fil)
    else:
        trajs = sorted([f for f in files if f.endswith(".traj")])

    if not trajs:
        print(f"错误: 当前目录 {cdir} 中没有找到匹配的 .traj 文件")
        print(f"  当前目录文件数: {len(files)}")
        if args.t:
            print(f"  过滤前缀: {args.t}")
        sys.exit(1)

    print(f"找到 {len(trajs)} 个 trajectory 文件:")
    for t in trajs:
        print(f"  - {t}")
    print(f"输出文件: {args.o}")
    print()

    total_frames = 0
    skipped = 0

    with open(args.o, "w") as f:
        for traj_name in trajs:
            # print("处理: {:s} ...\r".format(traj_name), end=" ", flush=True)
            images = Trajectory(traj_name)
            n_frames = 0

            for atoms in images:
                # 确定 config_type
                if args.config_type == "auto":
                    # 从 traj 文件名提取 (去掉 .traj 后缀)
                    config_type = traj_name.replace(".traj", "")
                elif args.config_type == "traj":
                    config_type = traj_name
                else:
                    config_type = args.config_type

                # 检查力数据
                forces = atoms.get_forces()
                if args.skip_no_force and forces is None:
                    skipped += 1
                    continue

                write_extxyz(f, atoms, n_frames, config_type=config_type)
                n_frames += 1

            print(f"处理: {traj_name} {n_frames} 帧 \r",end="\r", flush=True)
            total_frames += n_frames

    print(f"\n完成: 共写入 {total_frames} 帧到 {args.o}")
    if skipped:
        print(f"  跳过了 {skipped} 帧 (无 forces)")
    print(f"\n下一步: gap_fit atoms_filename={args.o} ...")


if __name__ == "__main__":
    main()
