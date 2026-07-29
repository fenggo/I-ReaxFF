#!/usr/bin/env python3
"""
汇总所有 results44-* 文件夹中 density.log 的 DFT 计算结果
散点图: Density vs Energy, 按文件夹着色, 标注结构ID
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

base = os.path.dirname(os.path.abspath(__file__))

# ── 收集所有 density.log ──
folders = sorted(glob.glob(os.path.join(base, 'results44-*')),
                 key=lambda x: int(x.split('-')[-1]))

data = {}  # folder_name -> (ids, densities, energies)
all_d, all_e, all_label = [], [], []

for folder in folders:
    fname = os.path.join(folder, 'density.log')
    if not os.path.isfile(fname):
        continue
    name = os.path.basename(folder)
    ids, ds, es = [], [], []
    with open(fname) as f:
        next(f)  # skip header
        for line in f:
            p = line.split()
            if len(p) >= 3:
                ids.append(int(p[0]))
                ds.append(float(p[1]))
                es.append(float(p[2]))
    if ids:
        data[name] = (ids, ds, es)
        all_d.extend(ds)
        all_e.extend(es)
        all_label.extend([name]*len(ds))

all_d = np.array(all_d)
all_e = np.array(all_e)
all_label = np.array(all_label)

print(f'\nTotal DFT points: {len(all_d)} from {len(data)} folders')
print(f'{"Folder":<16} {"Crystal_ID":>10} {"Density":>10} {"Energy":>16}')
print('-' * 56)
for name in sorted(data, key=lambda x: int(x.split('-')[-1])):
    ids, ds, es = data[name]
    for cid, d, e in zip(ids, ds, es):
        print(f'{name:<16} {cid:>10} {d:>10.4f} {e:>16.5f}')
print('-' * 56)
print(f'{"Total":<16} {len(all_d):>10}\n')

# ── 绘图 ──
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$Density$ ($g/cm^3$)', fontsize=13)
ax.set_ylabel(r'$Relative\ Energy$ ($eV$)', fontsize=13)
# ax.set_title(r'TNT$_4$·CL-20$_4$ DFT Results (4:4 molar ratio)', fontsize=13)

# 每个文件夹用不同颜色
cmap = plt.cm.tab10
n_folders = len(data)
folder_names = sorted(data.keys(), key=lambda x: int(x.split('-')[-1]))

for i, name in enumerate(folder_names):
    ids, ds, es = data[name]
    ax.scatter(ds, es, s=80, alpha=0.85, marker='o',
               color=cmap(i % 10), edgecolors='k', linewidths=0.5,
               label=name, zorder=3)
    # 标注结构ID
    for j in range(len(ids)):
        ax.annotate(str(ids[j]), (ds[j], es[j]), fontsize=6,
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points', color=cmap(i % 10))

# 标注最优结构
best_idx = np.argmin(all_e)
ax.scatter(all_d[best_idx], all_e[best_idx], marker='*', s=300,
           facecolors='none', edgecolors='red', linewidths=2,
           zorder=5, label=f'Best (D={all_d[best_idx]:.4f})')

# ── 边际 KDE ──
divider = make_axes_locatable(ax)
ax_top   = divider.append_axes("top",   size="15%", pad=0.18)
ax_right = divider.append_axes("right", size="15%", pad=0.08)

xl, yl = ax.get_xlim(), ax.get_ylim()

kde_d = gaussian_kde(all_d)
xd = np.linspace(xl[0], xl[1], 300)
ax_top.plot(xd, kde_d(xd), color='steelblue', lw=1.5)
ax_top.fill_between(xd, kde_d(xd), alpha=0.25, color='steelblue')
ax_top.set_xlim(xl); ax_top.set_xticks([]); ax_top.set_yticks([])
for sp in ['top','right','left']: ax_top.spines[sp].set_visible(False)

kde_e = gaussian_kde(all_e)
xe = np.linspace(yl[0], yl[1], 300)
ax_right.plot(kde_e(xe), xe, color='coral', lw=1.5)
ax_right.fill_between(kde_e(xe), xe, alpha=0.25, color='coral')
ax_right.set_ylim(yl); ax_right.set_xticks([]); ax_right.set_yticks([])
for sp in ['top','right','bottom']: ax_right.spines[sp].set_visible(False)

ax.legend(loc='lower left', fontsize=8, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.15)

# 信息框
# info = (f'Total DFT structures: {len(all_d)}\n'
#         f'Density: {all_d.min():.4f} – {all_d.max():.4f} g/cm³\n'
#         f'Energy: {all_e.min():.2f} – {all_e.max():.2f} eV\n'
#         f'Best: D={all_d[best_idx]:.4f}, E={all_e[best_idx]:.2f} eV')
# ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8,
#         verticalalignment='top', horizontalalignment='right',
#         bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

plt.savefig('dft_summary.png', dpi=200, bbox_inches='tight')
plt.savefig('dft_summary.svg', transparent=True, bbox_inches='tight')
print('\nSaved: dft_summary.png, dft_summary.svg')
