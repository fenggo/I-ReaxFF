"""JAX version of set_matrix_tensor.py.

In JAX the "parameters" are just a plain dict (pytree) of jnp arrays.
There is no nn.Parameter / nn.ParameterDict / nn.ParameterList -
trainability is controlled externally by optax (which params you pass
to the optimizer), not by a requires_grad flag.  We keep the same key
layout (e.g. 'fmwi_C', 'fmw_C' as a *list*, 'fmwo_C') so that the
energy/force code in reaxff_jax.py can index m identically.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _randn(shape, key):
    return jax.random.normal(key, shape, dtype=jnp.float64)


def get_universal_nn(spec, bonds, bo_universal_nn, be_universal,
                     vdw_universal_nn, mf_universal):
    universal_nn = []
    if bo_universal_nn is not None:
        universal_bonds = bonds if bo_universal_nn == 'all' else bo_universal_nn
        for bd in universal_bonds:
            b = bd.split('-')
            bdr = b[1] + '-' + b[0]
            universal_nn.append('fsi_' + bd)
            universal_nn.append('fpi_' + bd)
            universal_nn.append('fpp_' + bd)
            universal_nn.append('fsi_' + bdr)
            universal_nn.append('fpi_' + bdr)
            universal_nn.append('fpp_' + bdr)

    if be_universal is not None:
        universal_bonds = bonds if be_universal == 'all' else be_universal
        for bd in universal_bonds:
            b = bd.split('-')
            bdr = b[1] + '-' + b[0]
            universal_nn.append('fe_' + bd)
            universal_nn.append('fe_' + bdr)

    if vdw_universal_nn is not None:
        universal_bonds = bonds if vdw_universal_nn == 'all' else vdw_universal_nn
        for bd in universal_bonds:
            b = bd.split('-')
            bdr = b[1] + '-' + b[0]
            universal_nn.append('fv_' + bd)
            universal_nn.append('fv_' + bdr)

    if mf_universal is not None:
        universal_bonds = spec if mf_universal == 'all' else mf_universal
        for sp in universal_bonds:
            universal_nn.append('fm' + '_' + sp)
    return universal_nn


def set_matrix(m_, spec, bonds, mfopt, beopt, bdopt, messages,
               bo_layer, bo_layer_, BOFunction_, BOFunction,
               mf_layer, mf_layer_, MessageFunction_, MessageFunction,
               be_layer, be_layer_, EnergyFunction_, EnergyFunction,
               vdw_layer, vdw_layer_, VdwFunction_, VdwFunction,
               bo_universal_nn, be_universal, mf_universal, vdw_universal_nn,
               device='cpu', seed=0):
    ''' set variables for neural networks (pure dict of jnp arrays). '''
    # enable float64 so results match the torch.double version
    jax.config.update('jax_enable_x64', True)

    m = {}
    key = jax.random.PRNGKey(seed)
    rng = jax.random.key(seed)

    if mfopt is None:
        mfopt = spec
    if bdopt is None:
        bdopt = bonds
    if beopt is None:
        beopt = bonds

    universal_nn = get_universal_nn(spec, bonds, bo_universal_nn, be_universal,
                                    vdw_universal_nn, mf_universal)

    def split(rng):
        rng, k = jax.random.split(rng)
        return rng, k

    def set_wb(m_, pref='f', reuse_m=True, nin=8, nout=3, layer=[8, 9],
               vlist=None, bias=0.0):
        nonlocal rng
        for bd in vlist:
            if pref + '_' + bd in universal_nn:
                m[pref + 'wi_' + bd] = m[pref + 'wi']
                m[pref + 'bi_' + bd] = m[pref + 'bi']
            elif pref + 'wi_' + bd in m_ and reuse_m:
                arr_wi = np.array(m_[pref + 'wi_' + bd], dtype=np.float64)
                arr_bi = np.array(m_[pref + 'bi_' + bd], dtype=np.float64)
                m[pref + 'wi_' + bd] = jnp.array(arr_wi)
                m[pref + 'bi_' + bd] = jnp.array(arr_bi)
            else:
                rng, k1, k2 = split(rng), *jax.random.split(rng, 2)
                rng = jax.random.split(rng, 1)[0]
                k1, k2 = jax.random.split(jax.random.PRNGKey(seed + hash(bd) % 1000))
                m[pref + 'wi_' + bd] = _randn((nin, layer[0]), k1)
                m[pref + 'bi_' + bd] = _randn((layer[0],), k2)

            m[pref + 'w_' + bd] = []
            m[pref + 'b_' + bd] = []
            if pref + '_' + bd in universal_nn:
                m[pref + 'w_' + bd] = m[pref + 'w']
                m[pref + 'b_' + bd] = m[pref + 'b']
            elif pref + 'w_' + bd in m_ and reuse_m:
                for i in range(layer[1]):
                    arr_w = np.array(m_[pref + 'w_' + bd][i], dtype=np.float64)
                    arr_b = np.array(m_[pref + 'b_' + bd][i], dtype=np.float64)
                    m[pref + 'w_' + bd].append(jnp.array(arr_w))
                    m[pref + 'b_' + bd].append(jnp.array(arr_b))
            else:
                for i in range(layer[1]):
                    k1, k2 = jax.random.split(jax.random.PRNGKey(seed + i + hash(bd) % 1000), 2)
                    m[pref + 'w_' + bd].append(_randn((layer[0], layer[0]), k1))
                    m[pref + 'b_' + bd].append(_randn((layer[0],), k2))

            if pref + '_' + bd in universal_nn:
                m[pref + 'wo_' + bd] = m[pref + 'wo']
                m[pref + 'bo_' + bd] = m[pref + 'bo']
            elif pref + 'wo_' + bd in m_ and reuse_m:
                arr_wo = np.array(m_[pref + 'wo_' + bd], dtype=np.float64)
                arr_bo = np.array(m_[pref + 'bo_' + bd], dtype=np.float64)
                m[pref + 'wo_' + bd] = jnp.array(arr_wo)
                m[pref + 'bo_' + bd] = jnp.array(arr_bo)
            else:
                k1, k2 = jax.random.split(jax.random.PRNGKey(seed + 7 + hash(bd) % 1000), 2)
                m[pref + 'wo_' + bd] = _randn((layer[0], nout), k1) * 0.2
                m[pref + 'bo_' + bd] = _randn((nout,), k2) * 0.01 + bias

    def set_message_wb(m_, pref='f', reuse_m=True, nin=8, nout=3,
                       layer=[8, 9], bias=0.0):
        if m_ is None:
            m_ = {}
        for sp in spec:
            m[pref + 'w_' + sp] = []
            m[pref + 'b_' + sp] = []
            if pref + '_' + sp in universal_nn:
                m[pref + 'wi_' + sp] = m[pref + 'wi']
                m[pref + 'bi_' + sp] = m[pref + 'bi']
                m[pref + 'wo_' + sp] = m[pref + 'wo']
                m[pref + 'bo_' + sp] = m[pref + 'bo']
                m[pref + 'w_' + sp] = m[pref + 'w']
                m[pref + 'b_' + sp] = m[pref + 'b']
            elif pref + 'wi_' + sp in m_ and reuse_m:
                arr_wi = np.array(m_[pref + 'wi_' + sp], dtype=np.float64)
                arr_bi = np.array(m_[pref + 'bi_' + sp], dtype=np.float64)
                arr_wo = np.array(m_[pref + 'wo_' + sp], dtype=np.float64)
                arr_bo = np.array(m_[pref + 'bo_' + sp], dtype=np.float64)
                m[pref + 'wi_' + sp] = jnp.array(arr_wi)
                m[pref + 'bi_' + sp] = jnp.array(arr_bi)
                m[pref + 'wo_' + sp] = jnp.array(arr_wo)
                m[pref + 'bo_' + sp] = jnp.array(arr_bo)
                for i in range(layer[1]):
                    arr_w = np.array(m_[pref + 'w_' + sp][i], dtype=np.float64)
                    arr_b = np.array(m_[pref + 'b_' + sp][i], dtype=np.float64)
                    m[pref + 'w_' + sp].append(jnp.array(arr_w))
                    m[pref + 'b_' + sp].append(jnp.array(arr_b))
            else:
                k1, k2 = jax.random.split(jax.random.PRNGKey(seed + hash(sp) % 1000), 2)
                m[pref + 'wi_' + sp] = _randn((nin, layer[0]), k1)
                m[pref + 'bi_' + sp] = _randn((layer[0],), k2)
                k3, k4 = jax.random.split(jax.random.PRNGKey(seed + 3 + hash(sp) % 1000), 2)
                m[pref + 'wo_' + sp] = _randn((layer[0], nout), k3)
                m[pref + 'bo_' + sp] = _randn((nout,), k4)
                for i in range(layer[1]):
                    k1, k2 = jax.random.split(jax.random.PRNGKey(seed + i + hash(sp) % 1000), 2)
                    m[pref + 'w_' + sp].append(_randn((layer[0], layer[0]), k1))
                    m[pref + 'b_' + sp].append(_randn((layer[0],), k2))

    def set_universal_wb(m_, pref='f', bd='C-C', reuse_m=True, nin=8, nout=3,
                         layer=[8, 9], bias=0.0):
        if m_ is None:
            m_ = {}
        m[pref + 'w'] = []
        m[pref + 'b'] = []

        if pref + 'wi' in m_:
            bd_ = ''
        else:
            bd_ = '_' + bd

        if reuse_m and pref + 'wi' + bd_ in m_:
            arr_wi = np.array(m_[pref + 'wi' + bd_], dtype=np.float64)
            arr_bi = np.array(m_[pref + 'bi' + bd_], dtype=np.float64)
            arr_wo = np.array(m_[pref + 'wo' + bd_], dtype=np.float64)
            arr_bo = np.array(m_[pref + 'bo' + bd_], dtype=np.float64)
            m[pref + 'wi'] = jnp.array(arr_wi)
            m[pref + 'bi'] = jnp.array(arr_bi)
            m[pref + 'wo'] = jnp.array(arr_wo)
            m[pref + 'bo'] = jnp.array(arr_bo)
            for i in range(layer[1]):
                arr_w = np.array(m_[pref + 'w' + bd_][i], dtype=np.float64)
                arr_b = np.array(m_[pref + 'b' + bd_][i], dtype=np.float64)
                m[pref + 'w'].append(jnp.array(arr_w))
                m[pref + 'b'].append(jnp.array(arr_b))
        else:
            k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 2)
            m[pref + 'wi'] = _randn((nin, layer[0]), k1)
            m[pref + 'bi'] = _randn((layer[0],), k2)
            k3, k4 = jax.random.split(jax.random.PRNGKey(seed + 5), 2)
            m[pref + 'wo'] = _randn((layer[0], nout), k3)
            m[pref + 'bo'] = _randn((nout,), k4) + bias
            for i in range(layer[1]):
                k1, k2 = jax.random.split(jax.random.PRNGKey(seed + i), 2)
                m[pref + 'w'].append(_randn((layer[0], layer[0]), k1))
                m[pref + 'b'].append(_randn((layer[0],), k2))

    # ---- message NN ----
    if MessageFunction_ == 0 or (mf_layer == mf_layer_ and
                                 EnergyFunction == EnergyFunction_ and
                                 MessageFunction_ == MessageFunction):
        reuse_m = True
    else:
        reuse_m = False

    nout_ = 3 if MessageFunction != 4 else 1
    if MessageFunction == 1:
        nin_ = 7
    elif MessageFunction == 5:
        nin_ = 3
    else:
        nin_ = 3

    for t in range(1, messages + 1):
        b = 0.881373587 if t > 1 else -0.867
        if mf_universal is not None:
            set_universal_wb(m_=m_, pref='fm', bd=mf_universal[0],
                             reuse_m=reuse_m, nin=nin_, nout=nout_,
                             layer=mf_layer, bias=b)
        set_message_wb(m_=m_, pref='fm', reuse_m=reuse_m, nin=nin_,
                       nout=nout_, layer=mf_layer, bias=b)

    # ---- energy NN ----
    if EnergyFunction == EnergyFunction_ and be_layer == be_layer_:
        reuse_m = True
    else:
        reuse_m = False
    nin_ = 3

    if be_universal is not None:
        set_universal_wb(m_=m_, pref='fe', bd=be_universal[0],
                         reuse_m=reuse_m, nin=nin_, nout=1,
                         layer=be_layer, bias=2.0)
    set_wb(m_=m_, pref='fe', reuse_m=reuse_m, nin=nin_, nout=1,
           layer=be_layer, vlist=bonds, bias=2.0)

    return m
