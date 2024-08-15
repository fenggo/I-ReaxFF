import torch
from torch import nn

def set_matrix(m_,spec,bonds,mfopt,beopt,bdopt,messages,
               bo_layer,bo_layer_,BOFunction_,BOFunction,
               mf_layer,mf_layer_,MessageFunction_,MessageFunction,
               be_layer,be_layer_,EnergyFunction_,EnergyFunction,
               vdw_layer,vdw_layer_,VdwFunction_,VdwFunction,
               bo_universal_nn,be_universal,mf_universal,vdw_universal_nn,
               device='cpu'):
    ''' set variable for neural networks '''
    m = nn.ParameterDict() 
    bond   = []               # make sure the m matrix is unique 
    for si in spec:
        for sj in spec:
            bd = si + '-' + sj
            if bd not in bond:
                bond.append(bd)

    if mfopt is None:
       mfopt = spec
    if bdopt is None:
       bdopt = bonds
    if beopt is None:
       beopt = bonds

    universal_nn = get_universal_nn(spec,bonds,bo_universal_nn,be_universal,
                                    vdw_universal_nn,mf_universal)

    def set_wb(m_,pref='f',reuse_m=True,nin=8,nout=3,layer=[8,9],vlist=None,bias=0.0):
        ''' set matix varibles '''
        nonlocal m
        for bd in vlist:
            if pref+'_'+bd in universal_nn:
                m[pref+'wi_'+bd] = m[pref+'wi']
                m[pref+'bi_'+bd] = m[pref+'bi']
            elif pref+'wi_'+bd in m_ and reuse_m:                  # input layer
                if bd in bdopt:
                    m[pref+'wi_'+bd] = nn.Parameter(torch.tensor(m_[pref+'wi_'+bd],
                                             dtype=torch.double,device=device),requires_grad=True)
                    m[pref+'bi_'+bd] = nn.Parameter(torch.tensor(m_[pref+'bi_'+bd],
                                                 dtype=torch.double,device=device),requires_grad=True)
                else:
                    m[pref+'wi_'+bd] = torch.tensor(m_[pref+'wi_'+bd],dtype=torch.double,device=device)
                    m[pref+'bi_'+bd] = torch.tensor(m_[pref+'bi_'+bd],dtype=torch.double,device=device)
            else:
                m[pref+'wi_'+bd] = nn.Parameter(torch.randn(nin,layer[0],device=device),
                                                requires_grad=True)  
                m[pref+'bi_'+bd] = nn.Parameter(torch.randn(layer[0],device=device),
                                                    requires_grad=True)  

            m[pref+'w_'+bd] = nn.ParameterList()                                    # hidden layer
            m[pref+'b_'+bd] = nn.ParameterList()
            if pref+'_'+bd in universal_nn:
                for i in range(layer[1]):  
                    m[pref+'w_'+bd] = m[pref+'w']
                    m[pref+'b_'+bd] = m[pref+'b']
            elif pref+'w_'+bd in m_ and reuse_m:     
                if bd in bdopt:                            
                    for i in range(layer[1]):   
                        m[pref+'w_'+bd].append(nn.Parameter(torch.tensor(m_[pref+'w_'+bd][i],
                                                     dtype=torch.double,device=device),requires_grad=True) ) 
                        m[pref+'b_'+bd].append(nn.Parameter(torch.tensor(m_[pref+'b_'+bd][i],
                                                    dtype=torch.double,device=device),requires_grad=True) ) 
                else:
                    for i in range(layer[1]):   
                        m[pref+'w_'+bd].append(torch.tensor(m_[pref+'w_'+bd][i],dtype=torch.double,
                                            device=device) )
                        m[pref+'b_'+bd].append(torch.tensor(m_[pref+'b_'+bd][i],dtype=torch.double,
                                            device=device)) 
            else:
                for i in range(layer[1]):   
                    m[pref+'w_'+bd].append(nn.Parameter(torch.randn(layer[0],layer[0],device=device), 
                                                        requires_grad=True) ) 
                    m[pref+'b_'+bd].append(nn.Parameter(torch.randn(layer[0]),
                                                        requires_grad=True) ) 
            
            if pref+'_'+bd in universal_nn:
                m[pref+'wo_'+bd] = m[pref+'wo']
                m[pref+'bo_'+bd] = m[pref+'bo']
            elif pref+'wo_'+bd in m_ and reuse_m:          # output layer
                if bd in bdopt:       
                    m[pref+'wo_'+bd] = nn.Parameter(torch.tensor(m_[pref+'wo_'+bd],
                                             dtype=torch.double,device=device),requires_grad=True) 
                    m[pref+'bo_'+bd] = nn.Parameter(torch.tensor(m_[pref+'bo_'+bd],
                                             dtype=torch.double,device=device),requires_grad=True) 
                else:
                    m[pref+'wo_'+bd] = torch.tensor(m_[pref+'wo_'+bd],dtype=torch.double,device=device)
                    m[pref+'bo_'+bd] = torch.tensor(m_[pref+'bo_'+bd],dtype=torch.double,device=device)
            else:
                m[pref+'wo_'+bd] = nn.Parameter(torch.randn([layer[0],nout],stddev=0.2,device=device), 
                                                requires_grad=True)    
                m[pref+'bo_'+bd] = nn.Parameter(torch.randn([nout], stddev=0.01,device=device)+bias,
                                                    requires_grad=True) 
        return None

    def set_message_wb(m_,pref='f',reuse_m=True,nin=8,nout=3,layer=[8,9],bias=0.0):
        ''' set matix varibles '''
        nonlocal m
        if m_ is None:
           m_ = {}
        for sp in spec:
            m[pref+'w_'+sp] = nn.ParameterList()                                    
            m[pref+'b_'+sp] = nn.ParameterList()
            if pref+'_'+sp in universal_nn:
                m[pref+'wi_'+sp] = m[pref+'wi']
                m[pref+'bi_'+sp] = m[pref+'bi']
                m[pref+'wo_'+sp] = m[pref+'wo']
                m[pref+'bo_'+sp] = m[pref+'bo']
                m[pref+'w_'+sp]  = m[pref+'w'] 
                m[pref+'b_'+sp]  = m[pref+'b'] 
            elif pref+'wi_'+sp in m_ and reuse_m:
                if sp in mfopt:
                    m[pref+'wi_'+sp] = nn.Parameter(torch.tensor(m_[pref+'wi_'+sp],
                                            dtype=torch.double,device=device),requires_grad=True)
                    m[pref+'bi_'+sp] = nn.Parameter(torch.tensor(m_[pref+'bi_'+sp],
                                            dtype=torch.double,device=device), requires_grad=True)
                    m[pref+'wo_'+sp] = nn.Parameter(torch.tensor(m_[pref+'wo_'+sp],
                                            dtype=torch.double,device=device),requires_grad=True)
                    m[pref+'bo_'+sp] = nn.Parameter(torch.tensor(m_[pref+'bo_'+sp],
                                            dtype=torch.double,device=device),requires_grad=True)
                    for i in range(layer[1]):   
                        m[pref+'w_'+sp].append(nn.Parameter(torch.tensor(m_[pref+'w_'+sp][i],
                                            dtype=torch.double,device=device),requires_grad=True)) 
                        m[pref+'b_'+sp].append(nn.Parameter(torch.tensor(m_[pref+'b_'+sp][i],
                                            dtype=torch.double,device=device),requires_grad=True)) 
                else:
                    m[pref+'wi_'+sp] = torch.tensor(m_[pref+'wi_'+sp],dtype=torch.double,device=device)
                    m[pref+'bi_'+sp] = torch.tensor(m_[pref+'bi_'+sp],dtype=torch.double,device=device)
                    m[pref+'wo_'+sp] = torch.tensor(m_[pref+'wo_'+sp],dtype=torch.double,device=device)
                    m[pref+'bo_'+sp] = torch.tensor(m_[pref+'bo_'+sp],dtype=torch.double,device=device)
                    for i in range(layer[1]):   
                        m[pref+'w_'+sp].append(torch.tensor(m_[pref+'w_'+sp][i],
                                               dtype=torch.double,device=device ) ) 
                        m[pref+'b_'+sp].append(torch.tensor(m_[pref+'b_'+sp][i],
                                               dtype=torch.double,device=device ) ) 
            else:
                m[pref+'wi_'+sp] = nn.Parameter(torch.randn(nin,layer[0],device=device))   
                m[pref+'bi_'+sp] = nn.Parameter(torch.randn(layer[0],device=device))  
                m[pref+'wo_'+sp] = nn.Parameter(torch.randn(layer[0],nout,device=device))   
                m[pref+'bo_'+sp] = nn.Parameter(torch.randn(nout,device=device))  
                for i in range(layer[1]):   
                    m[pref+'w_'+sp].append(nn.Parameter(torch.randn(layer[0],layer[0],device=device ) ))
                    m[pref+'b_'+sp].append(nn.Parameter(torch.randn([layer[0]],device=device) )) 
        # return m  

    def set_universal_wb(m_,pref='f',bd='C-C',reuse_m=True,nin=8,nout=3,
                        layer=[8,9],bias=0.0):
        ''' set universial matix varibles '''
        nonlocal m
        if m_ is None:
           m_ = {}
        m[pref+'w'] = nn.ParameterList()                      # hidden layer
        m[pref+'b'] = nn.ParameterList()

        if pref+'wi' in m_:
           bd_ = ''
        else:
           bd_ = '_' + bd
        
        if reuse_m and pref+'wi'+bd_ in m_:   # input layer
            
            m[pref+'wi'] = nn.Parameter(torch.tensor(m_[pref+'wi'+bd_],
                                                    dtype=torch.double,device=device),requires_grad=True)
            m[pref+'bi'] = nn.Parameter(torch.tensor(m_[pref+'bi'+bd_],
                                                    dtype=torch.double,device=device),requires_grad=True)
            m[pref+'wo'] = nn.Parameter(torch.tensor(m_[pref+'wo'+bd_],
                                                    dtype=torch.double,device=device),requires_grad=True)
            m[pref+'bo'] = nn.Parameter(torch.tensor(m_[pref+'bo'+bd_],
                                                    dtype=torch.double,device=device),requires_grad=True)
            for i in range(layer[1]):   
                m[pref+'w'].append(nn.Parameter(torch.tensor(m_[pref+'w'+bd_][i],
                                                 dtype=torch.double,device=device),requires_grad=True) ) 
                m[pref+'b'].append(nn.Parameter(torch.tensor(m_[pref+'b'+bd_][i],
                                                 dtype=torch.double,device=device),requires_grad=True) )
        else:
            m[pref+'wi'] = nn.Parameter(torch.randn(nin,layer[0],device=device),requires_grad=True)   
            m[pref+'bi'] = nn.Parameter(torch.randn(layer[0],device=device),requires_grad=True)  
            m[pref+'wo'] = nn.Parameter(torch.randn(layer[0],nout,device=device),requires_grad=True)   
            m[pref+'bo'] = nn.Parameter(torch.randn(nout,device=device)+bias,requires_grad=True)
            for i in range(layer[1]):   
                m[pref+'w'].append(nn.Parameter(torch.randn(layer[0],layer[0],
                                        dtype=torch.double,device=device),requires_grad=True)) 
                m[pref+'b'].append(nn.Parameter(torch.randn(layer[0],
                                        dtype=torch.double,device=device),requires_grad=True)) 
        return None  # End of local funciton definition

    ############ set weight and bias for message neural network ###################
    if MessageFunction_==0 or (mf_layer==mf_layer_ and  EnergyFunction==EnergyFunction_
        and MessageFunction_==MessageFunction):
        reuse_m = True  
    else:
        reuse_m = False

    nout_ = 3 if MessageFunction!=4 else 1
    if MessageFunction==1:
        nin_  = 7
    elif MessageFunction==5 :
        nin_  = 3 
    else:
        nin_  = 3

    for t in range(1,messages+1):
        b = 0.881373587 if t>1 else -0.867
        if mf_universal is not None:
           set_universal_wb(m_=m_,pref='fm',bd=mf_universal[0],reuse_m=reuse_m, 
                            nin=nin_,nout=nout_,layer=mf_layer,bias=b)
        set_message_wb(m_=m_,pref='fm',reuse_m=reuse_m,nin=nin_,nout=nout_,   
                            layer=mf_layer,bias=b) 

    ############ set weight and bias for energy neural network ###################
    if EnergyFunction==EnergyFunction_ and be_layer==be_layer_:
        reuse_m = True  
    else:
        reuse_m = False 
    nin_ = 3 # 4 if EnergyFunction==1 else 3

    if not be_universal is None:
       set_universal_wb(m_=m_,pref='fe',bd=be_universal[0],reuse_m=reuse_m,
                        nin=nin_,nout=1,layer=be_layer, bias=2.0)
    set_wb(m_=m_,pref='fe',reuse_m=reuse_m,nin=nin_,nout=1,layer=be_layer,
           vlist=bonds,bias=2.0)

    nin_ = 1 if VdwFunction==1 else 3
    return m

def get_universal_nn(spec,bonds,bo_universal_nn,be_universal,vdw_universal_nn,mf_universal):
    universal_nn = []
    if not bo_universal_nn is None:
       if bo_universal_nn=='all':
          universal_bonds = bonds
       else:
          universal_bonds = bo_universal_nn
       for bd in universal_bonds:
           b = bd.split('-')
           bdr = b[1] + '-' + b[0]
           universal_nn.append('fsi_'+bd)
           universal_nn.append('fpi_'+bd)
           universal_nn.append('fpp_'+bd)
           universal_nn.append('fsi_'+bdr)
           universal_nn.append('fpi_'+bdr)
           universal_nn.append('fpp_'+bdr)

    if not be_universal is None:
       if be_universal=='all':
          universal_bonds = bonds
       else:
          universal_bonds = be_universal
       for bd in universal_bonds:
           b = bd.split('-')
           bdr = b[1] + '-' + b[0]
           universal_nn.append('fe_'+bd)
           universal_nn.append('fe_'+bdr)

    if not vdw_universal_nn is None:
       if vdw_universal_nn=='all':
          universal_bonds = bonds
       else:
          universal_bonds = vdw_universal_nn
       for bd in universal_bonds:
           b = bd.split('-')
           bdr = b[1] + '-' + b[0]
           universal_nn.append('fv_'+bd)
           universal_nn.append('fv_'+bdr)

    if not mf_universal is None:
       if mf_universal=='all':
          universal_bonds = spec
       else:
          universal_bonds = mf_universal
       for sp in universal_bonds:
           # for t in range(1,messages+1):
           universal_nn.append('fm'+'_'+sp) # +str(t)
    return universal_nn
