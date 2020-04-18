#!/usr/bin/env python
from __future__ import print_function
from ase import Atoms
from ase.io import read,write
import numpy as np
from .neighbors import get_neighbors,get_angle_p,get_torsion_p
from .reaxff_angle import get_eangle
import tensorflow as tf


def _tfvrtx(x, dx=np.ones(x.shape)):
    if not tangent.shapes_match(x, dx):
        raise ValueError(
            'Shape mismatch between argument value (%s) and seed derivative (%s)'
             % (numpy.shape(x), numpy.shape(dx)))

    # Primal and tangent of: xi = tf.expand_dims(x, 0)
    dxi = tf.expand_dims(dx, 0)

    # Primal and tangent of: xj = tf.expand_dims(x, 1)
    dxj = tf.expand_dims(dx, 1)

    # Primal and tangent of: vr = xi - xj
    dvr = dxi - dxj
    return dvr


def _tfrtvr(vr, dvr):
    ''' dbop/dvr '''

    # Primal and tangent of: vr2 = vr * vr
    tmp = vr * vr
    dvr2 = dvr * vr + vr * dvr
    vr2 = tmp

    # Primal and tangent of: np_sum_vr2 = np.sum(vr2, axis=2)
    tmp2 = np.sum(vr2, axis=2)
    dnp_sum_vr2 = numpy.sum(dvr2, axis=2, dtype=None, keepdims=False)
    np_sum_vr2 = tmp2

    # Primal and tangent of: R = np.sqrt(np_sum_vr2)
    tmp3 = np.sqrt(np_sum_vr2)
    dR = dnp_sum_vr2 / (2 * tmp3)
    return dR


def _tfboptr(r, ro, bo1=None, bo2=None, botol=None, dr=None):
    dbo1 = tangent.init_grad(bo1)
    dbotol = tangent.init_grad(botol)
    if not tangent.shapes_match(r, dr):
        raise ValueError(
            'Shape mismatch between argument value (%s) and seed derivative (%s)'
             % (numpy.shape(r), numpy.shape(dr)))

    # Primal and tangent of: bodiv = r / rosi
    tmp = r / rosi
    dbodiv = (dr * rosi - r * drosi) / (rosi * rosi)
    bodiv = tmp

    # Primal and tangent of: bopow = bodiv ** bo2
    tmp2 = bodiv ** bo2
    dbopow = bo2 * bodiv ** (bo2 - 1.0) * dbodiv
    bopow = tmp2

    # Primal and tangent of: bo1_times_bopow = bo1 * bopow
    tmp3 = bo1 * bopow
    _b9f0 = dbo1 * bopow + bo1 * dbopow
    bo1_times_bopow = tmp3

    # Primal and tangent of: _eterm = np.exp(bo1_times_bopow)
    tmp4 = np.exp(bo1_times_bopow)
    d_eterm = _b9f0 * tmp4
    _eterm = tmp4

    # Primal and tangent of: eterm = botol * _eterm
    determ = dbotol * _eterm + botol * d_eterm

    # Primal and tangent of: bop = eterm - botol
    dbop = determ - dbotol
    return dbop


def _tfdeltatbop(bop, dbop):
    # Primal and tangent of: deltap = np.sum(bop, axis=1)
    ddeltap = np.sum(dbop, axis=1)
    return ddeltap


