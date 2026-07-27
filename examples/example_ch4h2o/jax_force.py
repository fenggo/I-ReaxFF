# JAX-based force computation module
# Add these functions to reaxff_jax.py

import jax
import jax.numpy as jnp

# ── JAX versions of module-level functions ──

def taper_jnp(r, rmin=0.001, rmax=0.002):
    ''' Taper function for bond-order (JAX). '''
    r3 = jnp.where(r > rmax, jnp.ones_like(r), jnp.zeros_like(r))
    ok = jnp.logical_and(r <= rmax, r > rmin)
    r2 = jnp.where(ok, r, jnp.zeros_like(r))
    r20 = jnp.where(ok, jnp.ones_like(r), jnp.zeros_like(r))
    rterm = 1.0 / (rmin - rmax)**3.0
    rm = rmin * r20
    rd = rm - r2
    trm1 = rm + 2.0 * r2 - 3.0 * rmax * r20
    return rterm * rd * rd * trm1 + r3

def rtaper_jnp(r, rmin=0.001, rmax=0.002):
    ''' Reverse taper function (JAX). '''
    r3 = jnp.where(r < rmin, jnp.ones_like(r), jnp.zeros_like(r))
    ok = jnp.logical_and(r <= rmax, r > rmin)
    r2 = jnp.where(ok, r, jnp.zeros_like(r))
    r20 = jnp.where(ok, jnp.ones_like(r), jnp.zeros_like(r))
    rterm = 1.0 / (rmax - rmin)**3.0
    rm = rmax * r20
    rd = rm - r2
    trm1 = rm + 2.0 * r2 - 3.0 * rmin * r20
    return rterm * rd * rd * trm1 + r3

def fvr_jnp(x):
    ''' Pairwise vectors (JAX). '''
    xi = jnp.expand_dims(x, 1)
    xj = jnp.expand_dims(x, 2)
    return xj - xi

def fmessage_jnp(pre, bd, x, m, layer=5):
    ''' Message function NN (JAX). '''
    X = jnp.expand_dims(jnp.stack(x, axis=2), axis=2)
    o = [jax.nn.sigmoid(jnp.matmul(X, m[pre + 'wi_' + bd]) + m[pre + 'bi_' + bd])]
    for l in range(layer):
        o.append(jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'w_' + bd][l]) + m[pre + 'b_' + bd][l]))
    out = jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'wo_' + bd]) + m[pre + 'bo_' + bd])
    return jnp.squeeze(out, axis=2)

def fnn_jnp(pre, bd, x, m, layer=5):
    ''' Energy NN (JAX). '''
    X = jnp.expand_dims(jnp.stack(x, axis=2), axis=2)
    o = [jax.nn.sigmoid(jnp.matmul(X, m[pre + 'wi_' + bd]) + m[pre + 'bi_' + bd])]
    for l in range(layer):
        o.append(jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'w_' + bd][l]) + m[pre + 'b_' + bd][l]))
    out = jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'wo_' + bd]) + m[pre + 'bo_' + bd])
    return jnp.squeeze(out, axis=(2, 3))


def _compute_bond_energy_jax(st, x, rcell, cell, bdid, bonds, nbd, b, p, m,
                              botol, EnergyFunction, mf_layer, be_layer,
                              messages, acut, hbtol, log_):
    """Pure JAX function: x -> bond energy."""
    nbatch = x.shape[0]
    natom = x.shape[1]
    
    # 1. Pairwise vectors and distances
    vr = fvr_jnp(x)                              # (batch, n, n, 3)
    vrf = jnp.matmul(vr, rcell)                   # fractional coords
    vrf = jnp.where(vrf - 0.5 > 0, vrf - 1.0, vrf)
    vrf = jnp.where(vrf + 0.5 < 0, vrf + 1.0, vrf)
    vr = jnp.matmul(vrf, cell)                    # real coords
    r = jnp.sqrt(jnp.sum(vr * vr, axis=3) + 1e-8)  # distances
    
    # 2. Bond order computation
    bdid_full = bdid
    r_bond = r[:, bdid_full[:, 0], bdid_full[:, 1]]
    
    bop_si = jnp.zeros_like(r)
    bop_pi = jnp.zeros_like(r)
    bop_pp = jnp.zeros_like(r)
    
    for bd in bonds:
        if nbd.get(bd, 0) == 0:
            continue
        b_start, b_end = b[bd]
        ndx = bdid_full[b_start:b_end]
        bi = ndx[:, 0]
        bj = ndx[:, 1]
        
        r_bd = r[:, bi, bj]
        
        rr = log_ / p['bo1_' + bd]
        rc_bo = p['rosi_' + bd] * jnp.power(rr, 1.0 / p['bo2_' + bd])
        
        frc = jnp.where(jnp.logical_or(r_bd > rc_bo, r_bd <= 0.001), 0.0, 1.0)
        
        bodiv1 = r_bd / p['rosi_' + bd]
        bopow1 = jnp.power(bodiv1, p['bo2_' + bd])
        eterm1 = (1.0 + botol) * jnp.exp(p['bo1_' + bd] * bopow1) * frc
        
        bodiv2 = r_bd / p['ropi_' + bd]
        bopow2 = jnp.power(bodiv2, p['bo4_' + bd])
        eterm2 = jnp.exp(p['bo3_' + bd] * bopow2) * frc
        
        bodiv3 = r_bd / p['ropp_' + bd]
        bopow3 = jnp.power(bodiv3, p['bo6_' + bd])
        eterm3 = jnp.exp(p['bo5_' + bd] * bopow3) * frc
        
        bsi = taper_jnp(eterm1, rmin=botol, rmax=2.0 * botol) * (eterm1 - botol)
        bpi = taper_jnp(eterm2, rmin=botol, rmax=2.0 * botol) * eterm2
        bpp = taper_jnp(eterm3, rmin=botol, rmax=2.0 * botol) * eterm3
        
        bop_si = bop_si.at[:, bi, bj].set(bsi)
        bop_si = bop_si.at[:, bj, bi].set(bsi)
        bop_pi = bop_pi.at[:, bi, bj].set(bpi)
        bop_pi = bop_pi.at[:, bj, bi].set(bpi)
        bop_pp = bop_pp.at[:, bi, bj].set(bpp)
        bop_pp = bop_pp.at[:, bj, bi].set(bpp)
    
    bop = bop_si + bop_pi + bop_pp
    
    # 3. Message passing
    H = [bop]
    Hsi = [bop_si]
    Hpi = [bop_pi]
    Hpp = [bop_pp]
    D = [jnp.sum(bop, axis=2)]
    eye = jnp.expand_dims(1.0 - jnp.eye(natom), axis=0)
    
    for t in range(1, messages + 1):
        Di = jnp.expand_dims(D[t - 1], 2) * eye
        Dj = jnp.expand_dims(D[t - 1], 1) * eye
        Dbi = Di - H[t - 1]
        Dbj = Dj - H[t - 1]
        
        Dbi_ = Dbi[:, bdid_full[:, 0], bdid_full[:, 1]]
        Dbj_ = Dbj[:, bdid_full[:, 0], bdid_full[:, 1]]
        H_ = H[t - 1][:, bdid_full[:, 0], bdid_full[:, 1]]
        Hsi_ = Hsi[t - 1][:, bdid_full[:, 0], bdid_full[:, 1]]
        Hpi_ = Hpi[t - 1][:, bdid_full[:, 0], bdid_full[:, 1]]
        Hpp_ = Hpp[t - 1][:, bdid_full[:, 0], bdid_full[:, 1]]
        
        bo_new = jnp.zeros_like(r)
        bosi_new = jnp.zeros_like(r)
        bopi_new = jnp.zeros_like(r)
        bopp_new = jnp.zeros_like(r)
        
        for bd in bonds:
            if nbd.get(bd, 0) == 0:
                continue
            b_start, b_end = b[bd]
            ndx = bdid_full[b_start:b_end]
            bi = ndx[:, 0]
            bj = ndx[:, 1]
            bd_parts = bd.split('-')
            atom_i = bd_parts[0]
            atom_j = bd_parts[1]
            
            h = H_[:, b_start:b_end]
            hsi = Hsi_[:, b_start:b_end]
            hpi = Hpi_[:, b_start:b_end]
            hpp = Hpp_[:, b_start:b_end]
            dbi = Dbi_[:, b_start:b_end]
            dbj = Dbj_[:, b_start:b_end]
            
            Fi = fmessage_jnp('fm', atom_i, [dbi, h, dbj], m, layer=mf_layer[1])
            Fj = fmessage_jnp('fm', atom_j, [dbj, h, dbi], m, layer=mf_layer[1])
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
        
        H.append(bo_new)
        Hsi.append(bosi_new)
        Hpi.append(bopi_new)
        Hpp.append(bopp_new)
        D.append(jnp.sum(bo_new, axis=2))
    
    bo0 = H[-1]
    bosi = Hsi[-1]
    bopi = Hpi[-1]
    bopp = Hpp[-1]
    
    # 4. Bond energy
    bo = jnp.maximum(0, bo0 - acut)
    bosi_bond = bosi[:, bdid_full[:, 0], bdid_full[:, 1]]
    bopi_bond = bopi[:, bdid_full[:, 0], bdid_full[:, 1]]
    bopp_bond = bopp[:, bdid_full[:, 0], bdid_full[:, 1]]
    
    ebd = jnp.zeros_like(r)
    for bd in bonds:
        if nbd.get(bd, 0) == 0:
            continue
        b_start, b_end = b[bd]
        ndx = bdid_full[b_start:b_end]
        bi = ndx[:, 0]
        bj = ndx[:, 1]
        
        bosi_ = bosi_bond[:, b_start:b_end]
        bopi_ = bopi_bond[:, b_start:b_end]
        bopp_ = bopp_bond[:, b_start:b_end]
        
        if EnergyFunction == 0:
            FBO = jnp.where(bosi_ > 0.0, 1.0, 0.0)
            FBOR = 1.0 - FBO
            powb = jnp.power(bosi_ + FBOR, p['be2_' + bd])
            expb = jnp.exp(p['be1_' + bd] * (1.0 - powb))
            sieng = p['Desi_' + bd] * bosi_ * expb * FBO
            pieng = p['Depi_' + bd] * bopi_
            ppeng = p['Depp_' + bd] * bopp_
            esi = sieng + pieng + ppeng
        else:
            esi = fnn_jnp('fe', bd, [bosi_, bopi_, bopp_], m, layer=be_layer[1])
            esi = p['Desi_' + bd] * esi
        
        ebd = ebd.at[:, bi, bj].set(-esi)
        ebd = ebd.at[:, bj, bi].set(-esi)
    
    ebond = jnp.sum(ebd, axis=(1, 2))
    return ebond


def _compute_total_energy_jax(st, x, rcell, cell, bdid, bonds, nbd, b, p, m,
                               botol, EnergyFunction, mf_layer, be_layer,
                               messages, acut, hbtol, log_,
                               spec, s, p_ang, angs, na, a, ang_i, ang_j, ang_k,
                               ntor, t, tor_i, tor_j, tor_k, tor_l, tors,
                               opt_term, batch, vdwcut, hbshort, hblong,
                               hbs, nhb, hb_i, hb_j, hb_k,
                               dft_energy, dft_forces, weight_force, weight_energy,
                               q, eself, zpe):
    """Pure JAX function: x -> total energy."""
    ebond = _compute_bond_energy_jax(st, x, rcell, cell, bdid, bonds, nbd, b, p, m,
                                      botol, EnergyFunction, mf_layer, be_layer,
                                      messages, acut, hbtol, log_)
    return jnp.sum(ebond)