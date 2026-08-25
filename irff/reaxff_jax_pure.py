#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
irff/reaxff_jax_pure.py — Pure JAX ReaxFF energy computation.

Refactored from irff/reaxff_jax.py to be fully differentiable via jax.grad.
All energy components (bond, atomic, angle, torsion, VDW, HB, penalty) are
implemented as pure JAX functions with no side effects.

Architecture:
  - PureReaxFF: class that loads/precomputes static per-structure data
  - compute_energy(params, static, x): pure function → total energy
  - compute_loss(params, static, x, e_dft, f_dft): pure function → loss
  - jax.grad gives parameter gradients in ONE backward pass
"""

import json as js
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

# Enable float64
jax.config.update('jax_enable_x64', True)

# ═══════════════════════════════════════════════════════════════════
# Pure JAX helper functions
# ═══════════════════════════════════════════════════════════════════

def taper_jnp(r, rmin=0.001, rmax=0.002):
    """Taper function for bond-order cutoffs."""
    r3 = jnp.where(r > rmax, jnp.ones_like(r), jnp.zeros_like(r))
    ok = jnp.logical_and(r <= rmax, r > rmin)
    r2 = jnp.where(ok, r, jnp.zeros_like(r))
    r20 = jnp.where(ok, jnp.ones_like(r), jnp.zeros_like(r))
    rterm = 1.0 / (rmin - rmax) ** 3.0
    rm = rmin * r20
    rd = rm - r2
    trm1 = rm + 2.0 * r2 - 3.0 * rmax * r20
    return rterm * rd * rd * trm1 + r3


def rtaper_jnp(r, rmin=0.001, rmax=0.002):
    """Reverse taper function."""
    r3 = jnp.where(r < rmin, jnp.ones_like(r), jnp.zeros_like(r))
    ok = jnp.logical_and(r <= rmax, r > rmin)
    r2 = jnp.where(ok, r, jnp.zeros_like(r))
    r20 = jnp.where(ok, jnp.ones_like(r), jnp.zeros_like(r))
    rterm = 1.0 / (rmax - rmin) ** 3.0
    rm = rmax * r20
    rd = rm - r2
    trm1 = rm + 2.0 * r2 - 3.0 * rmin * r20
    return rterm * rd * rd * trm1 + r3


def fvr_jnp(x):
    """Pairwise vector differences: x_j - x_i."""
    xi = jnp.expand_dims(x, 1)
    xj = jnp.expand_dims(x, 2)
    return xj - xi


def sigmoid_jnp(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def div_safe(y, x):
    """Safe division: y/x, returns 0 where x==0."""
    return jnp.where(x != 0.0, y / x, jnp.zeros_like(y))


def div_safe_inf(y, x):
    """Safe division returning large value where x==0."""
    safe_x = jnp.where(x != 0.0, x, jnp.full_like(x, 1e-8))
    return jnp.where(x != 0.0, y / safe_x, y / safe_x)


# ═══════════════════════════════════════════════════════════════════
# Neural network message / energy functions (pure JAX)
# ═══════════════════════════════════════════════════════════════════

def fmessage_jnp(pre, bd, x_list, m, layer=5):
    """Message function NN: [Dbi, H, Dbj] → [Fsi, Fpi, Fpp]."""
    X = jnp.expand_dims(jnp.stack(x_list, axis=2), axis=2)
    o = [jax.nn.sigmoid(jnp.matmul(X, m[pre + 'wi_' + bd]) + m[pre + 'bi_' + bd])]
    for l in range(layer):
        o.append(jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'w_' + bd][l]) + m[pre + 'b_' + bd][l]))
    out = jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'wo_' + bd]) + m[pre + 'bo_' + bd])
    return jnp.squeeze(out, axis=2)


def fnn_jnp(pre, bd, x_list, m, layer=5):
    """Energy NN: [bosi, bopi, bopp] → scalar energy factor."""
    X = jnp.expand_dims(jnp.stack(x_list, axis=2), axis=2)
    o = [jax.nn.sigmoid(jnp.matmul(X, m[pre + 'wi_' + bd]) + m[pre + 'bi_' + bd])]
    for l in range(layer):
        o.append(jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'w_' + bd][l]) + m[pre + 'b_' + bd][l]))
    out = jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'wo_' + bd]) + m[pre + 'bo_' + bd])
    return jnp.squeeze(out, axis=(2, 3))


# ═══════════════════════════════════════════════════════════════════
# Bond order & message passing
# ═══════════════════════════════════════════════════════════════════

def compute_bond_orders(p, static, r):
    """Compute initial bond orders (si, pi, pp) from distances."""
    natom = static['natom']
    bdid = static['bdid']          # [nbd, 2]
    bond_names = static['bond_names']
    b_start_end = static['b_start_end']  # {name: (start, end)}
    nbd = static['nbd']            # {name: count}

    botol = p['botol']
    log_ = p['log_']

    bop_si = jnp.zeros_like(r)
    bop_pi = jnp.zeros_like(r)
    bop_pp = jnp.zeros_like(r)

    for bd_name in bond_names:
        if nbd.get(bd_name, 0) == 0:
            continue
        b_start, b_end = b_start_end[bd_name]
        ndx = bdid[b_start:b_end]
        bi = ndx[:, 0]
        bj = ndx[:, 1]
        r_bd = r[:, bi, bj]

        p_key = 'bo1_' + bd_name
        rr = log_ / p[p_key]
        rc_bo = p['rosi_' + bd_name] * jnp.power(rr, 1.0 / p['bo2_' + bd_name])
        frc = jnp.where(jnp.logical_or(r_bd > rc_bo, r_bd <= 0.001), 0.0, 1.0)

        bodiv1 = r_bd / p['rosi_' + bd_name]
        bopow1 = jnp.power(bodiv1, p['bo2_' + bd_name])
        eterm1 = (1.0 + botol) * jnp.exp(p['bo1_' + bd_name] * bopow1) * frc

        bodiv2 = r_bd / p['ropi_' + bd_name]
        bopow2 = jnp.power(bodiv2, p['bo4_' + bd_name])
        eterm2 = jnp.exp(p['bo3_' + bd_name] * bopow2) * frc

        bodiv3 = r_bd / p['ropp_' + bd_name]
        bopow3 = jnp.power(bodiv3, p['bo6_' + bd_name])
        eterm3 = jnp.exp(p['bo5_' + bd_name] * bopow3) * frc

        bsi = taper_jnp(eterm1, rmin=botol, rmax=2.0 * botol) * (eterm1 - botol)
        bpi = taper_jnp(eterm2, rmin=botol, rmax=2.0 * botol) * eterm2
        bpp = taper_jnp(eterm3, rmin=botol, rmax=2.0 * botol) * eterm3

        bop_si = bop_si.at[:, bi, bj].set(bsi)
        bop_si = bop_si.at[:, bj, bi].set(bsi)
        bop_pi = bop_pi.at[:, bi, bj].set(bpi)
        bop_pi = bop_pi.at[:, bj, bi].set(bpi)
        bop_pp = bop_pp.at[:, bi, bj].set(bpp)
        bop_pp = bop_pp.at[:, bj, bi].set(bpp)

    return bop_si, bop_pi, bop_pp


def message_passing(p, m, static, bop_si, bop_pi, bop_pp, r):
    """Iterative message passing to refine bond orders."""
    natom = static['natom']
    bdid = static['bdid']
    bond_names = static['bond_names']
    b_start_end = static['b_start_end']
    nbd = static['nbd']
    messages = static['messages']
    mf_layer = static['mf_layer']
    spec = static['spec']

    bop = bop_si + bop_pi + bop_pp
    H_list = [bop]
    Hsi_list = [bop_si]
    Hpi_list = [bop_pi]
    Hpp_list = [bop_pp]
    D_list = [jnp.sum(bop, axis=2)]

    eye = jnp.expand_dims(1.0 - jnp.eye(natom), axis=0)

    for t in range(1, messages + 1):
        Di = jnp.expand_dims(D_list[t - 1], 2) * eye
        Dj = jnp.expand_dims(D_list[t - 1], 1) * eye
        Dbi = Di - H_list[t - 1]
        Dbj = Dj - H_list[t - 1]

        Dbi_ = Dbi[:, bdid[:, 0], bdid[:, 1]]
        Dbj_ = Dbj[:, bdid[:, 0], bdid[:, 1]]
        H_ = H_list[t - 1][:, bdid[:, 0], bdid[:, 1]]
        Hsi_ = Hsi_list[t - 1][:, bdid[:, 0], bdid[:, 1]]
        Hpi_ = Hpi_list[t - 1][:, bdid[:, 0], bdid[:, 1]]
        Hpp_ = Hpp_list[t - 1][:, bdid[:, 0], bdid[:, 1]]

        bo_new = jnp.zeros_like(r)
        bosi_new = jnp.zeros_like(r)
        bopi_new = jnp.zeros_like(r)
        bopp_new = jnp.zeros_like(r)

        for bd_name in bond_names:
            if nbd.get(bd_name, 0) == 0:
                continue
            b_start, b_end = b_start_end[bd_name]
            ndx = bdid[b_start:b_end]
            bi = ndx[:, 0]
            bj = ndx[:, 1]
            bd_parts = bd_name.split('-')

            h = H_[:, b_start:b_end]
            hsi = Hsi_[:, b_start:b_end]
            hpi = Hpi_[:, b_start:b_end]
            hpp = Hpp_[:, b_start:b_end]
            dbi = Dbi_[:, b_start:b_end]
            dbj = Dbj_[:, b_start:b_end]

            Fi = fmessage_jnp('fm', bd_parts[0], [dbi, h, dbj], m, layer=mf_layer[1])
            Fj = fmessage_jnp('fm', bd_parts[1], [dbj, h, dbi], m, layer=mf_layer[1])
            F = Fi * Fj

            Fsi, Fpi, Fpp = F[..., 0], F[..., 1], F[..., 2]

            bo_new = bo_new.at[:, bi, bj].set(hsi * Fsi)
            bo_new = bo_new.at[:, bj, bi].set(hsi * Fsi)
            bosi_new = bosi_new.at[:, bi, bj].set(hsi * Fsi)
            bosi_new = bosi_new.at[:, bj, bi].set(hsi * Fsi)
            bopi_new = bopi_new.at[:, bi, bj].set(hpi * Fpi)
            bopi_new = bopi_new.at[:, bj, bi].set(hpi * Fpi)
            bopp_new = bopp_new.at[:, bi, bj].set(hpp * Fpp)
            bopp_new = bopp_new.at[:, bj, bi].set(hpp * Fpp)

        H_list.append(bo_new)
        Hsi_list.append(bosi_new)
        Hpi_list.append(bopi_new)
        Hpp_list.append(bopp_new)
        D_list.append(jnp.sum(bo_new, axis=2))

    return H_list, Hsi_list, Hpi_list, Hpp_list, D_list


def compute_bond_energy(p, m, static, bosi, bopi, bopp):
    """Compute bond energy from final bond orders."""
    bdid = static['bdid']
    bond_names = static['bond_names']
    b_start_end = static['b_start_end']
    nbd = static['nbd']
    be_layer = static['be_layer']
    EnergyFunction = static['EnergyFunction']

    natom = static['natom']
    nbatch = bosi.shape[0]
    ebd = jnp.zeros((nbatch, natom, natom))

    bosi_bond = bosi[:, bdid[:, 0], bdid[:, 1]]
    bopi_bond = bopi[:, bdid[:, 0], bdid[:, 1]]
    bopp_bond = bopp[:, bdid[:, 0], bdid[:, 1]]

    for bd_name in bond_names:
        if nbd.get(bd_name, 0) == 0:
            continue
        b_start, b_end = b_start_end[bd_name]
        ndx = bdid[b_start:b_end]
        bi = ndx[:, 0]
        bj = ndx[:, 1]

        bosi_ = bosi_bond[:, b_start:b_end]
        bopi_ = bopi_bond[:, b_start:b_end]
        bopp_ = bopp_bond[:, b_start:b_end]

        if EnergyFunction == 0:
            FBO = jnp.where(bosi_ > 0.0, 1.0, 0.0)
            FBOR = 1.0 - FBO
            powb = jnp.power(bosi_ + FBOR, p['be2_' + bd_name])
            expb = jnp.exp(p['be1_' + bd_name] * (1.0 - powb))
            sieng = p['Desi_' + bd_name] * bosi_ * expb * FBO
            pieng = p['Depi_' + bd_name] * bopi_
            ppeng = p['Depp_' + bd_name] * bopp_
            esi = sieng + pieng + ppeng
        else:
            esi = fnn_jnp('fe', bd_name, [bosi_, bopi_, bopp_], m, layer=be_layer[1])
            esi = p['Desi_' + bd_name] * esi

        ebd = ebd.at[:, bi, bj].set(-esi)
        ebd = ebd.at[:, bj, bi].set(-esi)

    return ebd, jnp.sum(ebd, axis=(1, 2))


# ═══════════════════════════════════════════════════════════════════
# Atomic energy (elone, eover, eunder)
# ═══════════════════════════════════════════════════════════════════

def compute_atomic_energy(p, static, Delta, Delta_pi, SO):
    """Compute lone-pair, over-coordination, under-coordination energies."""
    spec = static['spec']
    s = static['s']          # {sp: index array}
    ns = static['ns']        # {sp: count}
    nbatch = Delta.shape[0]
    natom = Delta.shape[1]

    Elone = jnp.zeros_like(Delta)
    Eover = jnp.zeros_like(Delta)
    Eunder = jnp.zeros_like(Delta)
    Nlp = jnp.zeros_like(Delta)
    Dlp = jnp.zeros_like(Delta)

    # elone
    for sp in spec:
        if ns.get(sp, 0) == 0:
            continue
        idx = s[sp]
        delta_sp = Delta[:, idx]

        Nlp_sp = 0.5 * (p['vale_' + sp] - p['val_' + sp])
        delta_e = 0.5 * (delta_sp - p['vale_' + sp])
        De = -jnp.maximum(0.0, -jnp.ceil(delta_e))
        nlp = -De + jnp.exp(-p['lp1'] * 4.0 * jnp.square(1.0 + delta_e - De))

        delta_lp = Nlp_sp - nlp
        dlp = delta_sp - p['val_' + sp] - delta_lp

        explp = 1.0 + jnp.exp(-75.0 * delta_lp)
        elone = p['lp2_' + sp] * delta_lp / explp

        Nlp = Nlp.at[:, idx].set(nlp)
        Dlp = Dlp.at[:, idx].set(dlp)
        Elone = Elone.at[:, idx].set(elone)

    # eover
    for sp in spec:
        if ns.get(sp, 0) == 0:
            continue
        idx = s[sp]
        delta_sp = Delta[:, idx]
        delta_lp_sp = 0.5 * (p['vale_' + sp] - p['val_' + sp]) - Nlp[:, idx]

        dpi = jnp.sum(Delta_pi[:, idx] * jnp.expand_dims(Dlp, 1)[:, idx], axis=2)

        delta_val = delta_sp - p['val_' + sp]
        delta_lpcorr = delta_val - delta_lp_sp / (1.0 + p['ovun3'] * jnp.exp(p['ovun4'] * dpi))
        otrm1 = div_safe_inf(1.0, delta_lpcorr + p['val_' + sp])
        otrm2 = sigmoid_jnp(-p['ovun2_' + sp] * delta_lpcorr)
        eover = SO[:, idx] * otrm1 * delta_lpcorr * otrm2

        Eover = Eover.at[:, idx].set(eover)

    # eunder
    for sp in spec:
        if ns.get(sp, 0) == 0:
            continue
        idx = s[sp]
        delta_sp = Delta[:, idx]
        delta_lp_sp = 0.5 * (p['vale_' + sp] - p['val_' + sp]) - Nlp[:, idx]
        delta_val = delta_sp - p['val_' + sp]
        dpi = jnp.sum(Delta_pi[:, idx] * jnp.expand_dims(Dlp, 1)[:, idx], axis=2)
        delta_lpcorr = delta_val - delta_lp_sp / (1.0 + p['ovun3'] * jnp.exp(p['ovun4'] * dpi))

        expeu1 = jnp.exp(p['ovun6'] * delta_lpcorr)
        eu1 = sigmoid_jnp(p['ovun2_' + sp] * delta_lpcorr)
        expeu3 = jnp.exp(p['ovun8'] * dpi)
        eu2 = 1.0 / (1.0 + p['ovun7'] * expeu3)
        eunder = -p['ovun5_' + sp] * (1.0 - expeu1) * eu1 * eu2

        Eunder = Eunder.at[:, idx].set(eunder)

    elone = jnp.sum(Elone, axis=1)
    eover = jnp.sum(Eover, axis=1)
    eunder = jnp.sum(Eunder, axis=1)

    # eatomic
    eatomic = jnp.zeros(nbatch)
    for sp in spec:
        if ns.get(sp, 0) > 0:
            eatomic = eatomic - p['atomic_' + sp] * ns[sp]

    return elone, eover, eunder, eatomic


# ═══════════════════════════════════════════════════════════════════
# Three-body energy (angle, penalty, conjugation)
# ═══════════════════════════════════════════════════════════════════

def compute_threebody_energy(p, static, bo, bopi, Delta, Delta_ang, Nlp, vr, fbot, delta_pi):
    """Compute angle, penalty, and three-body conjugation energies."""
    ang_names = static['ang_names']
    a_start_end = static['a_start_end']
    na = static['na']
    ang_i = static['ang_i']
    ang_j = static['ang_j']
    ang_k = static['ang_k']
    spec = static['spec']
    nbatch = bo.shape[0]

    nang = static['nang']
    if nang == 0:
        return (jnp.zeros(nbatch), jnp.zeros(nbatch), jnp.zeros(nbatch),
                jnp.zeros(nbatch), jnp.zeros(nbatch), jnp.zeros(nbatch))

    eang_total = jnp.zeros(nbatch)
    epen_total = jnp.zeros(nbatch)
    etcon_total = jnp.zeros(nbatch)

    for ang in ang_names:
        if na.get(ang, 0) == 0:
            continue
        sp = ang.split('-')[1]
        a0, a1 = a_start_end[ang]
        ai = ang_i[a0:a1]
        aj = ang_j[a0:a1]
        ak = ang_k[a0:a1]

        boij = bo[:, ai, aj]
        bojk = bo[:, aj, ak]
        fij = fbot[:, ai, aj]
        fjk = fbot[:, aj, ak]
        fijk = fij * fjk

        delta_j = Delta[:, aj]
        delta_ang_j = Delta_ang[:, aj]
        delta_i = Delta[:, ai]
        delta_k = Delta[:, ak]
        sbo = delta_pi[:, aj]
        nlp_j = Nlp[:, aj]

        # PBO
        PBOpow = -jnp.power(bo + 1e-8, 8)
        PBOexp = jnp.exp(PBOpow)
        Pbo = jnp.prod(PBOexp, axis=2)
        pbo_j = Pbo[:, aj]

        # theta
        theta = compute_theta(static, vr, ai, aj, ak)

        # f7
        Fboi = jnp.where(boij > 0.0, 1.0, 0.0)
        Fbori = 1.0 - Fboi
        expij = jnp.exp(-p['val3_' + sp] * jnp.power(boij + Fbori, p['val4_' + ang]) * Fboi)
        Fbok = jnp.where(bojk > 0.0, 1.0, 0.0)
        Fbork = 1.0 - Fbok
        expjk = jnp.exp(-p['val3_' + sp] * jnp.power(bojk + Fbork, p['val4_' + ang]) * Fbok)
        f7 = (1.0 - expij) * (1.0 - expjk)

        # f8
        exp6 = jnp.exp(p['val6'] * delta_ang_j)
        exp7 = jnp.exp(-p['val7_' + ang] * delta_ang_j)
        f8 = p['val5_' + sp] - (p['val5_' + sp] - 1.0) * (2.0 + exp6) / (1.0 + exp6 + exp7)

        # theta0
        Sbo = sbo - (1.0 - pbo_j) * (delta_ang_j + p['val8'] * nlp_j)
        S1 = jnp.where(jnp.logical_and(Sbo <= 1.0, Sbo > 0.0), Sbo, 0.0)
        Sbo1 = jnp.where(jnp.logical_and(Sbo <= 1.0, Sbo > 0.0),
                         jnp.power(S1 + 1e-8, p['val9']), 0.0)
        S2 = jnp.where(jnp.logical_and(Sbo < 2.0, Sbo > 1.0), Sbo, 0.0)
        F2 = jnp.where(jnp.logical_and(Sbo < 2.0, Sbo > 1.0), 1.0, 0.0)
        S2 = 2.0 * F2 - S2
        Sbo12 = jnp.where(jnp.logical_and(Sbo < 2.0, Sbo > 1.0),
                          2.0 - jnp.power(S2 + 1e-8, p['val9']), 0.0)
        Sbo2 = jnp.where(Sbo >= 2.0, 1.0, 0.0)
        Sbo3 = Sbo1 + Sbo12 + 2.0 * Sbo2
        theta0_deg = 180.0 - p['theta0_' + ang] * (1.0 - jnp.exp(-p['val10'] * (2.0 - Sbo3)))
        theta0 = theta0_deg / 57.29577951

        thet = theta0 - theta
        thet2 = jnp.square(thet)
        expang = jnp.exp(-p['val2_' + ang] * thet2)
        Eang = fijk * f7 * f8 * (p['val1_' + ang] - p['val1_' + ang] * expang)

        # penalty
        f9 = (2.0 + jnp.exp(-p['pen3'] * delta_j)) / (1.0 + jnp.exp(-p['pen3'] * delta_j) + jnp.exp(p['pen4'] * delta_j))
        expi = jnp.exp(-p['pen2'] * jnp.square(boij - 2.0))
        expk = jnp.exp(-p['pen2'] * jnp.square(bojk - 2.0))
        Epen = p['pen1_' + ang] * f9 * expi * expk * fijk

        # three-body conjugation
        delta_coa = delta_ang_j
        expcoa1 = jnp.exp(p['coa2'] * delta_coa)
        texp0 = p['coa1_' + ang] / (1.0 + expcoa1)
        texp1 = jnp.exp(-p['coa3'] * jnp.square(delta_i - boij))
        texp2 = jnp.exp(-p['coa3'] * jnp.square(delta_k - bojk))
        texp3 = jnp.exp(-p['coa4'] * jnp.square(boij - 1.5))
        texp4 = jnp.exp(-p['coa4'] * jnp.square(bojk - 1.5))
        Etcon = texp0 * texp1 * texp2 * texp3 * texp4 * fijk

        eang_total = eang_total + jnp.sum(Eang, axis=1)
        epen_total = epen_total + jnp.sum(Epen, axis=1)
        etcon_total = etcon_total + jnp.sum(Etcon, axis=1)

    return eang_total, epen_total, etcon_total, jnp.zeros(nbatch), jnp.zeros(nbatch), jnp.zeros(nbatch)


def compute_theta(static, vr, ai, aj, ak):
    """Compute bond angle theta from pairwise vectors."""
    Rij = jnp.sqrt(jnp.sum(jnp.square(vr[:, ai, aj]), axis=2))
    Rjk = jnp.sqrt(jnp.sum(jnp.square(vr[:, aj, ak]), axis=2))
    vik = vr[:, ai, aj] + vr[:, aj, ak]
    Rik = jnp.sqrt(jnp.sum(jnp.square(vik), axis=2))

    Rij2 = Rij * Rij
    Rjk2 = Rjk * Rjk
    Rik2 = Rik * Rik

    cos_theta = (Rij2 + Rjk2 - Rik2) / (2.0 * Rij * Rjk)
    cos_theta = jnp.clip(cos_theta, -0.9999999, 0.9999999)
    return jnp.arccos(cos_theta)


# ═══════════════════════════════════════════════════════════════════
# Four-body energy (torsion, conjugation)
# ═══════════════════════════════════════════════════════════════════

def compute_fourbody_energy(p, static, bo, bopi, Delta_ang, vr, fbot):
    """Compute torsion and four-body conjugation energies."""
    tor_names = static['tor_names']
    t_start_end = static['t_start_end']
    nt = static['nt']
    tor_i = static['tor_i']
    tor_j = static['tor_j']
    tor_k = static['tor_k']
    tor_l = static['tor_l']
    nbatch = bo.shape[0]

    ntor = static['ntor']
    if ntor == 0:
        return jnp.zeros(nbatch), jnp.zeros(nbatch)

    etor_total = jnp.zeros(nbatch)
    efcon_total = jnp.zeros(nbatch)

    for tor in tor_names:
        if nt.get(tor, 0) == 0:
            continue
        t0, t1 = t_start_end[tor]
        ti = tor_i[t0:t1]
        tj = tor_j[t0:t1]
        tk = tor_k[t0:t1]
        tl = tor_l[t0:t1]

        boij = bo[:, ti, tj]
        bojk = bo[:, tj, tk]
        bokl = bo[:, tk, tl]
        bopjk = bopi[:, tj, tk]
        fij = fbot[:, ti, tj]
        fjk = fbot[:, tj, tk]
        fkl = fbot[:, tk, tl]
        fijkl = fij * fjk * fkl

        delta_j = Delta_ang[:, tj]
        delta_k = Delta_ang[:, tk]

        # Torsion angle
        w, cos_w, cos2w, s_ijk, s_jkl = compute_torsion_angle(vr, ti, tj, tk, tl)

        # f10
        exp1 = 1.0 - jnp.exp(-p['tor2'] * boij)
        exp2 = 1.0 - jnp.exp(-p['tor2'] * bojk)
        exp3 = 1.0 - jnp.exp(-p['tor2'] * bokl)
        f10 = exp1 * exp2 * exp3

        # f11
        delt = delta_j + delta_k
        f11exp3 = jnp.exp(-p['tor3'] * delt)
        f11exp4 = jnp.exp(p['tor4'] * delt)
        f11 = (2.0 + f11exp3) / (1.0 + f11exp3 + f11exp4)

        expv2 = jnp.exp(p['tor1_' + tor] * jnp.square(2.0 - bopjk - f11))
        cos3w = jnp.cos(3.0 * w)
        v1 = 0.5 * p['V1_' + tor] * (1.0 + cos_w)
        v2 = 0.5 * p['V2_' + tor] * expv2 * (1.0 - cos2w)
        v3 = 0.5 * p['V3_' + tor] * (1.0 + cos3w)
        Etor = fijkl * f10 * s_ijk * s_jkl * (v1 + v2 + v3)

        # Four-body conjugation
        exptol = jnp.exp(-p['cot2'] * jnp.square(p['acut'] - 1.5))
        expij = jnp.exp(-p['cot2'] * jnp.square(boij - 1.5)) - exptol
        expjk = jnp.exp(-p['cot2'] * jnp.square(bojk - 1.5)) - exptol
        expkl = jnp.exp(-p['cot2'] * jnp.square(bokl - 1.5)) - exptol
        f12 = expij * expjk * expkl
        prod = 1.0 + (jnp.square(jnp.cos(w)) - 1.0) * s_ijk * s_jkl
        Efcon = fijkl * f12 * p['cot1_' + tor] * prod

        etor_total = etor_total + jnp.sum(Etor, axis=1)
        efcon_total = efcon_total + jnp.sum(Efcon, axis=1)

    return etor_total, efcon_total


def compute_torsion_angle(vr, ti, tj, tk, tl):
    """Compute torsion angle omega and related quantities."""
    rij = jnp.sqrt(jnp.sum(jnp.square(vr[:, ti, tj]), axis=2))
    rjk = jnp.sqrt(jnp.sum(jnp.square(vr[:, tj, tk]), axis=2))
    rkl = jnp.sqrt(jnp.sum(jnp.square(vr[:, tk, tl]), axis=2))

    vrjk = vr[:, tj, tk]
    vrkl = vr[:, tk, tl]
    vrjl = vrjk + vrkl
    rjl = jnp.sqrt(jnp.sum(jnp.square(vrjl), axis=2))

    vrij = vr[:, ti, tj]
    vril = vrij + vrjl
    ril = jnp.sqrt(jnp.sum(jnp.square(vril), axis=2))

    vrik = vrij + vrjk
    rik = jnp.sqrt(jnp.sum(jnp.square(vrik), axis=2))

    rij2 = jnp.square(rij)
    rjk2 = jnp.square(rjk)
    rkl2 = jnp.square(rkl)
    rjl2 = jnp.square(rjl)
    ril2 = jnp.square(ril)
    rik2 = jnp.square(rik)

    c_ijk = (rij2 + rjk2 - rik2) / (2.0 * rij * rjk)
    c2ijk = jnp.square(c_ijk)
    cijk = 1.00000001 - c2ijk
    s_ijk = jnp.sqrt(cijk)

    c_jkl = (rjk2 + rkl2 - rjl2) / (2.0 * rjk * rkl)
    c2jkl = jnp.square(c_jkl)
    cjkl = 1.00000001 - c2jkl
    s_jkl = jnp.sqrt(cjkl)

    c_kjl = (rjk2 + rjl2 - rkl2) / (2.0 * rjk * rjl)
    c2kjl = jnp.square(c_kjl)
    ckjl = 1.00000001 - c2kjl
    s_kjl = jnp.sqrt(ckjl)

    fz = rij2 + rjl2 - ril2 - 2.0 * rij * rjl * c_ijk * c_kjl
    fm = rij * rjl * s_ijk * s_kjl

    fm = jnp.where(jnp.logical_and(fm <= 0.000001, fm >= -0.000001),
                   jnp.ones_like(fm), fm)
    fac = jnp.where(jnp.logical_and(fm <= 0.000001, fm >= -0.000001),
                    jnp.zeros_like(fm), jnp.ones_like(fm))
    cos_w = 0.5 * fz * fac / fm
    cos_w = jnp.clip(cos_w, -0.999999, 0.999999)
    w = jnp.arccos(cos_w)
    cos2w = jnp.cos(2.0 * w)

    return w, cos_w, cos2w, s_ijk, s_jkl


# ═══════════════════════════════════════════════════════════════════
# VDW + Coulomb energy
# ═══════════════════════════════════════════════════════════════════

def compute_vdw_coulomb_energy(p, static, vr, cell, q):
    """Compute van der Waals and Coulomb energies via Ewald-like sum."""
    nbatch = vr.shape[0]
    natom = vr.shape[1]
    vdwcut = static['vdwcut']

    # Build per-atom parameter arrays
    spec = static['spec']
    s = static['s']
    vdw1 = p['vdw1']

    # gamma and gammaw per atom
    gamma = jnp.zeros((nbatch, natom))
    gammaw = jnp.zeros((nbatch, natom))
    for sp in spec:
        if static['ns'].get(sp, 0) > 0:
            idx = s[sp]
            gamma = gamma.at[:, idx].set(p['gamma_' + sp])
            gammaw = gammaw.at[:, idx].set(p['gammaw_' + sp])

    # Devdw, alfa, rvdw per pair (using pmask-like approach)
    bond_names = static['bond_names']
    vb_i = static['vb_i']
    vb_j = static['vb_j']
    ns = static['ns']

    Devdw = jnp.zeros((nbatch, natom, natom))
    alfa = jnp.zeros((nbatch, natom, natom))
    rvdw = jnp.zeros((nbatch, natom, natom))
    for bd in bond_names:
        if not vb_i.get(bd):
            continue
        bi = jnp.array(vb_i[bd])
        bj = jnp.array(vb_j[bd])
        Devdw = Devdw.at[:, bi, bj].set(p['Devdw_' + bd])
        alfa = alfa.at[:, bi, bj].set(p['alfa_' + bd])
        rvdw = rvdw.at[:, bi, bj].set(p['rvdw_' + bd])

    gamma_pair = jnp.sqrt(jnp.expand_dims(gamma, 1) * jnp.expand_dims(gamma, 2))
    gm3 = jnp.power(1.0 / gamma_pair, 3.0)

    Evdw = jnp.zeros((nbatch, natom, natom))
    Ecoul = jnp.zeros((nbatch, natom, natom))

    cell0 = cell[:, :, 0]
    cell1 = cell[:, :, 1]
    cell2 = cell[:, :, 2]
    cell0 = jnp.expand_dims(cell0, 1)
    cell1 = jnp.expand_dims(cell1, 1)
    cell2 = jnp.expand_dims(cell2, 1)

    nc = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                cell_shift = cell0 * i + cell1 * j + cell2 * k
                vr_ = vr + cell_shift
                r = jnp.sqrt(jnp.sum(jnp.square(vr_), axis=3) + 1e-8)
                r3 = jnp.power(r + 1e-8, 3.0)

                fv_ = jnp.where(jnp.logical_and(r > 0.0000001, r <= vdwcut), 1.0, 0.0)
                if nc < 13:
                    fv = jnp.triu(fv_, k=0)
                else:
                    fv = jnp.triu(fv_, k=1)

                # f13
                gammaw_pair = jnp.sqrt(jnp.expand_dims(gammaw, 1) * jnp.expand_dims(gammaw, 2))
                rr = jnp.power(r, vdw1) + jnp.power(1.0 / gammaw_pair, vdw1)
                f13 = jnp.power(rr, 1.0 / vdw1)

                # Taper
                tpc = (1.0 + (-35.0 / vdwcut ** 4.0) * jnp.power(r, 4.0) +
                       (84.0 / vdwcut ** 5.0) * jnp.power(r, 5.0) +
                       (-70.0 / vdwcut ** 6.0) * jnp.power(r, 6.0) +
                       (20.0 / vdwcut ** 7.0) * jnp.power(r, 7.0))

                expvdw1 = jnp.exp(0.5 * alfa * (1.0 - f13 / (2.0 * rvdw)))
                expvdw2 = jnp.square(expvdw1)
                Evdw = Evdw + fv * tpc * Devdw * (expvdw2 - 2.0 * expvdw1)

                rth = jnp.power(r3 + gm3, 1.0 / 3.0)
                Ecoul = Ecoul + fv * tpc * q / rth
                nc += 1

    evdw = jnp.sum(Evdw, axis=(1, 2))
    ecoul = jnp.sum(Ecoul, axis=(1, 2))
    return evdw, ecoul


# ═══════════════════════════════════════════════════════════════════
# Hydrogen bond energy
# ═══════════════════════════════════════════════════════════════════

def compute_hb_energy(p, static, bo0, vr, cell, fhb):
    """Compute hydrogen bond energy."""
    hb_names = static['hb_names']
    h_start_end = static['h_start_end']
    nhb = static['nhb']
    hb_i = static['hb_i']
    hb_j = static['hb_j']
    hb_k = static['hb_k']
    nbatch = bo0.shape[0]
    hbshort = static['hbshort']
    hblong = static['hblong']

    ehb = jnp.zeros(nbatch)

    cell0 = jnp.expand_dims(cell[:, :, 0], 1)  # [nbatch, 1, 3]
    cell1 = jnp.expand_dims(cell[:, :, 1], 1)
    cell2 = jnp.expand_dims(cell[:, :, 2], 1)

    for hb in hb_names:
        if nhb.get(hb, 0) == 0:
            continue
        h0, h1 = h_start_end[hb]
        hi = hb_i[h0:h1]
        hj = hb_j[h0:h1]
        hk = hb_k[h0:h1]

        bo = bo0[:, hi, hj]
        fhb_ij = fhb[:, hi, hj]

        rij = jnp.sqrt(jnp.sum(jnp.square(vr[:, hi, hj]), axis=2))
        rij2 = jnp.square(rij)
        vrij = vr[:, hi, hj]
        vrjk_ = vr[:, hj, hk]  # [nbatch, n_pairs, 3]

        ehb_bd = jnp.zeros_like(bo)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    cell_shift = cell0 * i + cell1 * j + cell2 * k  # [nbatch, 1, 3]
                    vrjk = vrjk_ + cell_shift

                    rjk2 = jnp.sum(jnp.square(vrjk), axis=3)
                    rjk = jnp.sqrt(rjk2 + 1e-8)

                    vrik = vrij + vrjk
                    rik2 = jnp.sum(jnp.square(vrik), axis=3)
                    rik = jnp.sqrt(rik2 + 1e-8)

                    cos_th = (rij2 + rjk2 - rik2) / (2.0 * rij * rjk)
                    hbthe = 0.5 - 0.5 * cos_th
                    frhb = rtaper_jnp(rik, rmin=hbshort, rmax=hblong)

                    exphb1 = 1.0 - jnp.exp(-p['hb1_' + hb] * bo)
                    hbsum = p['rohb_' + hb] / rjk + rjk / p['rohb_' + hb] - 2.0
                    exphb2 = jnp.exp(-p['hb2_' + hb] * hbsum)

                    sin4 = jnp.square(hbthe)
                    ehb_bd = ehb_bd + fhb_ij * frhb * p['Dehb_' + hb] * exphb1 * exphb2 * sin4

        ehb = ehb + jnp.sum(ehb_bd, axis=(1, 2))

    return ehb


# ═══════════════════════════════════════════════════════════════════
# MASTER: total energy computation (pure JAX)
# ═══════════════════════════════════════════════════════════════════

def compute_energy(p, m, static, x):
    """
    Pure JAX function: (parameters, NN weights, static data, positions) → total energy.

    Args:
        p:  dict of scalar parameters (e.g. p['Desi_C-H'], p['val1_C-C-C'], ...)
        m:  dict of NN weight arrays (e.g. m['fmwi_C'], m['few_C-H'], ...)
        static: dict of pre-computed per-structure data
        x:  positions [nbatch, natom, 3]

    Returns:
        Total energy per batch [nbatch]
    """
    cell = static['cell']        # [nbatch, 3, 3]
    rcell = static['rcell']      # [nbatch, 3, 3]

    # 1. Pairwise vectors and distances (with PBC)
    vr = fvr_jnp(x)
    vrf = jnp.matmul(vr, rcell)
    vrf = jnp.where(vrf - 0.5 > 0, vrf - 1.0, vrf)
    vrf = jnp.where(vrf + 0.5 < 0, vrf + 1.0, vrf)
    vr = jnp.matmul(vrf, cell)
    r = jnp.sqrt(jnp.sum(vr * vr, axis=3) + 1e-8)

    # 2. Bond orders
    bop_si, bop_pi, bop_pp = compute_bond_orders(p, static, r)

    # 3. Message passing
    H_list, Hsi_list, Hpi_list, Hpp_list, D_list = message_passing(
        p, m, static, bop_si, bop_pi, bop_pp, r)

    # Final bond orders
    bosi = Hsi_list[-1]
    bopi = Hpi_list[-1]
    bopp = Hpp_list[-1]
    bo0 = H_list[-1]
    bo = jnp.maximum(0.0, bo0 - p['acut'])

    # Intermediate quantities
    Delta = D_list[-1]      # total bond order sum per atom
    Delta_pi = jnp.sum(bopi, axis=2)  # pi bond order sum
    # SO (same as in original code, depends on Delta_pi and p['ovun1'])
    SO = jnp.zeros_like(Delta)
    for bd_name in static['bond_names']:
        if static['nbd'].get(bd_name, 0) == 0:
            continue
        b_start, b_end = static['b_start_end'][bd_name]
        ndx = static['bdid'][b_start:b_end]
        bi = ndx[:, 0]
        bj = ndx[:, 1]
        bso = p['ovun1_' + bd_name] * p['Desi_' + bd_name] * bo0[:, bi, bj]
        SO = SO.at[:, bi].add(bso)
        SO = SO.at[:, bj].add(bso)
    SO = 0.5 * SO

    # fbot (bond order taper for angle/torsion)
    fbot = jnp.where(bo > 0.001, 1.0, 0.0)

    # 4. Bond energy
    ebd, ebond = compute_bond_energy(p, m, static, bosi, bopi, bopp)

    # 5. Atomic energy
    elone, eover, eunder, eatomic = compute_atomic_energy(p, static, Delta, Delta_pi, SO)

    # 6. Three-body energy
    Delta_ang = Delta  # will be refined in angle computation
    eang, epen, etcon, _, _, _ = compute_threebody_energy(
        p, static, bo, bopi, Delta, Delta_ang, jnp.zeros_like(Delta), vr, fbot, Delta_pi)

    # 7. Four-body energy
    etor, efcon = compute_fourbody_energy(p, static, bo, bopi, Delta_ang, vr, fbot)

    # 8. VDW + Coulomb
    q = static['q']  # charges
    evdw, ecoul = compute_vdw_coulomb_energy(p, static, vr, cell, q)

    # 9. HB energy
    ehb = compute_hb_energy(p, static, bo0, vr, cell, fbot)

    # 10. Self energy (q * (chi + q * mu))
    chi = jnp.zeros_like(Delta)
    mu = jnp.zeros_like(Delta)
    for sp in static['spec']:
        if static['ns'].get(sp, 0) > 0:
            idx = static['s'][sp]
            chi = chi.at[:, idx].set(p['chi_' + sp])
            mu = mu.at[:, idx].set(p['mu_' + sp])
    eself = jnp.sum(q * (chi + q * mu), axis=1)

    # 11. Molecular energy correction
    emol = static['emol']  # per-structure correction

    total = ebond + elone + eover + eunder + eatomic + eang + epen + etcon + etor + efcon + evdw + ecoul + ehb + eself + emol

    return total


# ═══════════════════════════════════════════════════════════════════
# PureReaxFF: data loading + static data precomputation
# ═══════════════════════════════════════════════════════════════════

class PureReaxFF:
    """
    Pure JAX ReaxFF wrapper.

    Loads dataset, precomputes static per-structure data as JAX arrays,
    builds parameter dicts (p, m) from ffield.json, and provides a
    pure loss function differentiable via jax.grad.

    Usage:
        model = PureReaxFF(dataset=..., libfile='ffield.json', ...)
        params, opt_state = model.init_optimizer(lr=1e-4)
        for step in range(n_steps):
            params, opt_state, loss = model.train_step(params, opt_state)
    """

    def __init__(self, dataset={}, data={},
                 libfile='ffield.json',
                 vdwcut=10.0,
                 hbshort=6.75, hblong=7.5,
                 EnergyFunction=1, MessageFunction=3,
                 mf_layer=None, be_layer=None,
                 cons=None,
                 clip={},
                 weight_force={'others': 1.0}, weight_energy={'others': 1.0},
                 bo_clip={},
                 lambda_bd=1000.0, lambda_reg=0.01,
                 lambda_pi=0.0, lambda_ang=0.0,
                 screen=False,
                 device={'all': 'cpu'}):

        self.dataset = dataset
        self.data = data
        self.vdwcut = vdwcut
        self.hbshort = hbshort
        self.hblong = hblong
        self.EnergyFunction = EnergyFunction
        self.MessageFunction = MessageFunction
        self.clip = clip
        self.bo_clip = bo_clip
        self.lambda_bd = lambda_bd
        self.lambda_reg = lambda_reg
        self.lambda_pi = lambda_pi
        self.lambda_ang = lambda_ang
        self.screen = screen
        self._device = device
        self.weight_force = weight_force
        self.weight_energy = weight_energy

        self.cons = cons if cons else []
        self.mf_layer = mf_layer
        self.be_layer = be_layer

        # Load ffield
        self._read_ffield(libfile)

        # Load data
        self._load_data()

        # Build static dicts per structure
        self._build_static()

        # Build parameter dicts
        self._build_params()

    def _read_ffield(self, libfile):
        """Read force field file."""
        if libfile.endswith('.json'):
            with open(libfile, 'r') as f:
                j = js.load(f)
            self.p_ = j['p']
            self.m_ = j['m']
            self.MolEnergy_ = j.get('MolEnergy', {})
            self.messages = j.get('messages', 1)
            self.BOFunction = j.get('BOFunction', 0)
            self.EnergyFunction_ = j.get('EnergyFunction', 0)
            self.MessageFunction_ = j.get('MessageFunction', 0)
            self.VdwFunction = j.get('VdwFunction', 0)
            self.mf_layer_ = j.get('mf_layer', [9, 2])
            self.be_layer_ = j.get('be_layer', [9, 1])
            self.rcut = j.get('rcut', {})
            self.rcuta = j.get('rcutBond', {})
            self.re = j.get('rEquilibrium', {})
            self.emol = 0.0
        else:
            from irff.reaxfflib import read_ffield as _read_ff
            (self.p_, _, self.spec, self.bonds, self.offd, self.angs,
             self.torp, self.hbs) = _read_ff(libfile=libfile, zpe=False)
            self.m_ = None
            self.mf_layer_ = None
            self.be_layer_ = None
            self.emol = 0.0
            self.rcut = None
            self.rcuta = None
            self.re = None
            self.EnergyFunction_ = 0
            self.MessageFunction_ = 0
            self.VdwFunction = 0
            self.p_['acut'] = 0.0001
            self.p_['hbtol'] = 0.0001

        if self.mf_layer is None:
            self.mf_layer = self.mf_layer_
        if self.be_layer is None:
            self.be_layer = self.be_layer_

        self._init_bonds()

    def _init_bonds(self):
        """Extract bond, angle, torsion, HB lists from p_."""
        self.bonds, self.offd, self.angs, self.torp, self.hbs = [], [], [], [], []
        self.spec = []
        for key in self.p_:
            k = key.split('_')
            if k[0] == 'bo1':
                self.bonds.append(k[1])
            elif k[0] == 'rosi':
                kk = k[1].split('-')
                if len(kk) == 2:
                    self.offd.append(k[1])
            elif k[0] == 'theta0':
                self.angs.append(k[1])
            elif k[0] == 'tor1':
                self.torp.append(k[1])
            elif k[0] == 'rohb':
                self.hbs.append(k[1])
            elif k[0] == 'val':
                self.spec.append(k[1])

        self.spec = sorted(set(self.spec))
        self.bonds = sorted(set(self.bonds))
        self.angs = sorted(set(self.angs))
        self.hbs = sorted(set(self.hbs))

        from irff.intCheck import check_tors
        self.tors = check_tors(self.spec, self.torp)

    def _load_data(self):
        """Load trajectory data using reax_force_data."""
        from irff.reax_force_data import reax_force_data

        self.strcs = []
        self._raw_data = {}

        for st in self.dataset:
            if st not in self.data:
                data_ = reax_force_data(
                    structure=st,
                    traj=self.dataset[st],
                    vdwcut=self.vdwcut,
                    rcut=self.rcut,
                    rcuta=self.rcuta,
                    hbshort=self.hbshort,
                    hblong=self.hblong,
                    variable_batch=True,
                    sample='uniform',
                    m=self.m_,
                    mf_layer=self.mf_layer_,
                    p=self.p_, spec=self.spec, bonds=self.bonds,
                    angs=self.angs, tors=self.tors,
                    hbs=self.hbs,
                    screen=self.screen)
                self.data[st] = data_
            else:
                data_ = self.data[st]

            if data_.status:
                self.strcs.append(st)
                self._raw_data[st] = data_

        self.nframe = sum(d.batch for d in self._raw_data.values())

    def _build_static(self):
        """Precompute static per-structure data as JAX arrays."""
        self.static = {}

        for st in self.strcs:
            d = self._raw_data[st]
            natom = d.natom

            static = {}

            # Basic info
            static['natom'] = natom
            static['nang'] = d.nang
            static['ntor'] = d.ntor
            static['messages'] = self.messages
            static['mf_layer'] = self.mf_layer
            static['be_layer'] = self.be_layer
            static['EnergyFunction'] = self.EnergyFunction
            static['vdwcut'] = self.vdwcut
            static['hbshort'] = self.hbshort
            static['hblong'] = self.hblong
            static['spec'] = self.spec

            # Positions, cell, rcell
            static['x'] = jnp.array(d.x)
            static['cell'] = jnp.array(jnp.expand_dims(d.cell, axis=1))
            static['rcell'] = jnp.array(jnp.expand_dims(d.rcell, axis=1))
            static['q'] = jnp.array(d.qij)

            # DFT references
            static['dft_energy'] = jnp.array(d.energy_dft)
            if d.forces is not None:
                static['dft_forces'] = jnp.array(d.forces)
            else:
                static['dft_forces'] = None

            # Bond indices
            static['bdid'] = jnp.array(d.bond)  # [nbd, 2]
            static['bond_names'] = self.bonds
            static['nbd'] = d.nbd

            b_start_end = {}
            for bd in self.bonds:
                b_start_end[bd] = d.B[bd]
            static['b_start_end'] = b_start_end

            # Angle indices
            static['ang_names'] = self.angs
            static['na'] = d.na
            static['ang_i'] = jnp.array(d.ang_i)
            static['ang_j'] = jnp.array(d.ang_j)
            static['ang_k'] = jnp.array(d.ang_k)
            a_start_end = {}
            for ang in self.angs:
                a_start_end[ang] = d.A.get(ang, (0, 0))
            # Only keep angles that exist in this structure
            static['ang_names'] = [ang for ang in self.angs if d.na.get(ang, 0) > 0]
            static['na'] = d.na
            static['ang_i'] = jnp.array(d.ang_i)
            static['ang_j'] = jnp.array(d.ang_j)
            static['ang_k'] = jnp.array(d.ang_k)
            static['a_start_end'] = a_start_end

            # Torsion indices
            static['tor_names'] = [tor for tor in self.tors if d.nt.get(tor, 0) > 0]
            static['nt'] = d.nt
            static['tor_i'] = jnp.array(d.tor_i)
            static['tor_j'] = jnp.array(d.tor_j)
            static['tor_k'] = jnp.array(d.tor_k)
            static['tor_l'] = jnp.array(d.tor_l)
            t_start_end = {}
            for tor in self.tors:
                t_start_end[tor] = d.T.get(tor, (0, 0))
            static['t_start_end'] = t_start_end

            # HB indices (flattened from dict to 1D arrays)
            hb_i_flat = []
            hb_j_flat = []
            hb_k_flat = []
            h_start_end = {}
            for hb in self.hbs:
                h_start = len(hb_i_flat)
                if hb in d.hb_i:
                    hb_i_flat.extend([x[0] for x in d.hb_i[hb]])
                    hb_j_flat.extend([x[0] for x in d.hb_j[hb]])
                    hb_k_flat.extend([x[0] for x in d.hb_k[hb]])
                h_end = len(hb_i_flat)
                h_start_end[hb] = (h_start, h_end)
            static['hb_i'] = jnp.array(hb_i_flat)
            static['hb_j'] = jnp.array(hb_j_flat)
            static['hb_k'] = jnp.array(hb_k_flat)
            static['h_start_end'] = h_start_end
            static['hb_names'] = [hb for hb in self.hbs if d.nhb.get(hb, 0) > 0]
            static['nhb'] = d.nhb

            # Species → atom index mapping
            s_dict = {}
            ns_dict = {}
            for sp in self.spec:
                idx = [i for i, name in enumerate(d.atom_name) if name == sp]
                s_dict[sp] = jnp.array(idx)
                ns_dict[sp] = len(idx)
            static['s'] = s_dict
            static['ns'] = ns_dict

            # vb_i, vb_j for VDW pair masks
            vb_i = {}
            vb_j = {}
            for i in range(natom):
                for j in range(natom):
                    bd = d.atom_name[i] + '-' + d.atom_name[j]
                    if bd not in self.bonds:
                        bd = d.atom_name[j] + '-' + d.atom_name[i]
                    if bd not in vb_i:
                        vb_i[bd] = []
                        vb_j[bd] = []
                    vb_i[bd].append(i)
                    vb_j[bd].append(j)
            static['vb_i'] = vb_i
            static['vb_j'] = vb_j

            # Molecular energy correction
            st_ = st.split('-')[0]
            if st_ in self.MolEnergy_:
                static['emol'] = self.MolEnergy_[st_]
            else:
                static['emol'] = 0.0

            self.static[st] = static

    def _build_params(self):
        """Build p (scalar params) and m (NN weights) dicts."""
        from irff.set_matrix_jax import set_matrix as _set_matrix
        from irff.intCheck import Intelligent_Check

        self.unit = 4.3364432032e-2
        self.punit = ['Desi', 'Depi', 'Depp', 'lp2', 'ovun5', 'val1',
                      'coa1', 'V1', 'V2', 'V3', 'cot1', 'pen1', 'Devdw', 'Dehb']

        p_bond = ['Desi', 'ovun1', 'Depi', 'Depp',
                  'bo3', 'bo4', 'bo1', 'bo2', 'bo5', 'bo6', 'be2', 'be1',
                  'Devdw', 'rvdw', 'alfa', 'rosi', 'ropi', 'ropp']
        p_offd = ['Devdw', 'rvdw', 'alfa', 'rosi', 'ropi', 'ropp']
        p_g = ['coa2', 'ovun6', 'lp1', 'lp3',
               'ovun7', 'ovun8', 'val6', 'tor2',
               'val8', 'val9', 'val10',
               'tor3', 'tor4', 'cot2', 'coa4', 'ovun4',
               'ovun3', 'val8', 'coa3', 'pen2', 'pen3', 'pen4',
               'acut', 'vdw1']
        p_spec = ['valang', 'valboc', 'val', 'vale',
                  'lp2', 'ovun5', 'val3', 'val5',
                  'ovun2', 'atomic',
                  'mass', 'chi', 'mu', 'gamma', 'gammaw']
        p_ang = ['theta0', 'val1', 'val2', 'coa1', 'val7', 'val4', 'pen1']
        p_hb = ['rohb', 'Dehb', 'hb1', 'hb2']
        p_tor = ['V1', 'V2', 'V3', 'tor1', 'cot1']

        ic = Intelligent_Check(re=self.re if self.re else {}, clip=self.clip, spec=self.spec,
                               bonds=self.bonds, offd=self.offd,
                               angs=self.angs, tors=self.torp, ptor=p_tor)
        self.p_, self.m_, cons = ic.check(self.p_, self.m_)

        self.botol = 0.01 * self.p_['cutoff']
        self.log_ = -9.21044036697651

        # Determine which params are trainable (opt)
        sp_opt = set()
        bd_opt = set()
        ang_opt = set()
        tor_opt = set()
        hb_opt = set()
        for sp in self.spec:
            for st in self.strcs:
                if self.static[st]['ns'].get(sp, 0) > 0:
                    sp_opt.add(sp)
                    break
        for bd in self.bonds:
            for st in self.strcs:
                if self.static[st]['nbd'].get(bd, 0) > 0:
                    bd_opt.add(bd)
                    break
        for ang in self.angs:
            for st in self.strcs:
                if self.static[st]['na'].get(ang, 0) > 0:
                    ang_opt.add(ang)
                    break
        for tor in self.tors:
            for st in self.strcs:
                if self.static[st]['nt'].get(tor, 0) > 0:
                    tor_opt.add(tor)
                    break
        for hb in self.hbs:
            for st in self.strcs:
                if self.static[st]['nhb'].get(hb, 0) > 0:
                    hb_opt.add(hb)
                    break

        opt = []
        for key in p_g:
            if key not in self.cons:
                opt.append(key)
        for key in p_spec:
            if key not in self.cons:
                opt.append(key)
        for key in p_bond:
            if key not in self.cons:
                opt.append(key)
        for key in p_ang:
            if key not in self.cons:
                opt.append(key)
        for key in p_tor:
            if key not in self.cons:
                opt.append(key)
        for key in p_hb:
            if key not in self.cons:
                opt.append(key)

        self.opt = opt

        # Build p dict
        p = {}
        pp = {}  # trainable subset

        # Global params
        for key in p_g:
            unit_ = self.unit if key in self.punit else 1.0
            val = self.p_[key] * unit_
            if key in opt:
                pp[key] = jnp.array(val)
                p[key] = pp[key]
            else:
                p[key] = jnp.array(val)

        p['acut'] = jnp.clip(p['acut'], self.p_['acut'] * 0.95, self.p_['acut'] * 1.05)

        # Species params
        for key in p_spec:
            unit_ = self.unit if key in self.punit else 1.0
            for sp in self.spec:
                key_ = key + '_' + sp
                val = self.p_.get(key_, 0.0) * unit_
                if (key in opt or key_ in opt) and sp in sp_opt:
                    pp[key_] = jnp.array(val)
                    p[key_] = pp[key_]
                else:
                    p[key_] = jnp.array(val)

        # Bond params
        for key in p_bond:
            unit_ = self.unit if key in self.punit else 1.0
            for bd in self.bonds:
                key_ = key + '_' + bd
                val = self.p_.get(key + '_' + bd, 0.0) * unit_
                if (key in opt or key_ in opt) and bd in bd_opt:
                    pp[key_] = jnp.array(val)
                    p[key_] = pp[key_]
                else:
                    p[key_] = jnp.array(val)

        # Angle params
        for key in p_ang:
            unit_ = self.unit if key in self.punit else 1.0
            for a in self.angs:
                key_ = key + '_' + a
                val = self.p_.get(key_, 0.0) * unit_
                if (key in opt or key_ in opt) and a in ang_opt:
                    pp[key_] = jnp.array(val)
                    p[key_] = pp[key_]
                else:
                    p[key_] = jnp.array(val)

        # Torsion params
        for key in p_tor:
            unit_ = self.unit if key in self.punit else 1.0
            for t in self.tors:
                key_ = key + '_' + t
                val = self.p_.get(key_, 0.0) * unit_
                if (key in opt or key_ in opt) and t in tor_opt:
                    pp[key_] = jnp.array(val)
                    p[key_] = pp[key_]
                else:
                    p[key_] = jnp.array(val)

        # HB params
        for key in p_hb:
            unit_ = self.unit if key in self.punit else 1.0
            for h in self.hbs:
                key_ = key + '_' + h
                val = self.p_.get(key_, 0.0) * unit_
                if (key in opt or key_ in opt) and h in hb_opt:
                    pp[key_] = jnp.array(val)
                    p[key_] = pp[key_]
                else:
                    p[key_] = jnp.array(val)

        # Add botol and log_ as constants
        p['botol'] = jnp.array(self.botol)
        p['log_'] = jnp.array(self.log_)

        # Build m (NN weights) via set_matrix
        m = _set_matrix(self.m_, self.spec, self.bonds,
                        None, None, None, 1,
                        (6, 0), (6, 0), 0, 0,
                        self.mf_layer, self.mf_layer_,
                        self.MessageFunction_, self.MessageFunction,
                        self.be_layer, self.be_layer_, 1, 1,
                        (9, 0), (9, 0), 1, 1,
                        None, None, None, None)

        self.p = p
        self.pp = pp
        self.m = m
        self.ic = ic

    def clamp_params(self, params):
        """Clamp parameters to physical bounds."""
        # params is a dict {key: float}
        for k in list(params.keys()):
            key = k.split('_')[0]
            unit_ = self.unit if key in self.punit else 1.0
            if k in self.ic.clip:
                lo = self.ic.clip[k][0] * unit_
                hi = self.ic.clip[k][1] * unit_
                params[k] = jnp.clip(params[k], lo, hi)
            elif key in self.ic.clip:
                lo = self.ic.clip[key][0] * unit_
                hi = self.ic.clip[key][1] * unit_
                params[k] = jnp.clip(params[k], lo, hi)
        return params

    # ═══════════════════════════════════════════════════════════════
    # Pure loss function (differentiable)
    # ═══════════════════════════════════════════════════════════════

    def _loss_fn(self, params, m, st):
        """Pure loss for a single structure. Returns (loss, energy_loss, force_loss)."""
        static = self.static[st]
        x = static['x']
        e_dft = static['dft_energy']
        f_dft = static['dft_forces']

        # Merge trainable params into full p
        p_full = dict(self.p)
        p_full.update(params)

        # Energy
        energy_fn = lambda x_: compute_energy(p_full, m, static, x_)
        e_pred = energy_fn(x)

        # Forces via grad
        force_fn = jax.grad(lambda x_: jnp.sum(energy_fn(x_)))
        f_pred = -force_fn(x)

        # Energy loss
        we = self.weight_energy.get(st, self.weight_energy.get('others', 1.0))
        loss_e = we * jnp.sum(jnp.square(e_pred - e_dft))

        # Force loss
        if f_dft is not None:
            wf_key = st.split('-')[0]
            wf = self.weight_force.get(st, self.weight_force.get(wf_key, 1.0))
            loss_f = wf * jnp.sum(jnp.square(f_pred - f_dft))
        else:
            loss_f = jnp.array(0.0)

        # Regularization penalty
        penalty = self._compute_penalty(p_full, m, static)

        return loss_e + loss_f + penalty, loss_e, loss_f

    def _compute_penalty(self, params, m, static):
        """Compute regularization penalty (simplified)."""
        penalty = jnp.array(0.0)
        if self.lambda_reg > 0.000001:
            # L2 on NN weights
            for k in m:
                if isinstance(m[k], list):
                    for w in m[k]:
                        penalty = penalty + jnp.sum(jnp.square(w))
                elif isinstance(m[k], jnp.ndarray):
                    if m[k].ndim >= 1:
                        penalty = penalty + jnp.sum(jnp.square(m[k]))
        penalty = penalty * self.lambda_reg

        # Bond-order penalty (simplified)
        if self.lambda_bd > 0.000001:
            for bd in static['bond_names']:
                if static['nbd'].get(bd, 0) == 0:
                    continue
                rr = params['log_'] / params['bo1_' + bd]
                rc_bo = params['rosi_' + bd] * jnp.power(rr, 1.0 / params['bo2_' + bd])
                rcut_bd = self.rcut.get(bd, 3.0)
                penalty = penalty + self.lambda_bd * jnp.maximum(0.0, rc_bo - rcut_bd)

        return penalty

    # ═══════════════════════════════════════════════════════════════
    # Training interface
    # ═══════════════════════════════════════════════════════════════

    def init_optimizer(self, lr=1e-4):
        """Initialize optax optimizer and parameter state."""
        import optax
        params = dict(self.pp)  # copy
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params)
        return params, optimizer, opt_state

    def make_step_fn(self):
        """Create a jitted training step function."""
        import optax

        optimizer = optax.adam(learning_rate=1e-4)  # placeholder, overridden

        def step_fn(params, opt_state, optimizer, st):
            """Single training step for one structure."""
            # Compute loss and gradient
            (loss, loss_e, loss_f), grads = jax.value_and_grad(
                lambda p: self._loss_fn(p, self.m, st), has_aux=True)(params)

            # Update
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self.clamp_params(params)

            return params, opt_state, loss, loss_e, loss_f

        return step_fn