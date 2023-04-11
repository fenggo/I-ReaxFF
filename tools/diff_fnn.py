import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
import matplotlib.pyplot as plt

wh = np.array([ [ 0.2976251244544983,
          1.6859064102172852,
          0.8486582636833191,
          -0.2851971983909607,
          -1.1658705472946167,
          1.0119858980178833 ],
        [ 0.10063588619232178,
          0.7935754656791687,
          -0.23131096363067627,
          -0.4436652958393097,
          0.45993709564208984,
          -0.02776309847831726 ],
        [  -1.4830913543701172,
          -0.025211017578840256,
          0.614945113658905,
          -2.086026668548584,
          0.24895654618740082,
          0.8985058665275574  ],
        [ 0.5602737665176392,
          0.824817419052124,
          -0.8687888383865356,
          0.9319312572479248,
          0.5129152536392212,
          0.4243922829627991  ],
        [
          0.010895919986069202,
          -1.6019028425216675,
          -1.5118945837020874,
          0.4212455153465271,
          -0.152587890625,
          0.08005668967962265  ],
        [ -0.2994173467159271,
          0.8387343287467957,
          -0.15869559347629547,
          0.5399070978164673,
          0.058497387915849686,
          0.6697747707366943  ]  ]  )
bh =  np.array([ -0.5495955348014832,
        1.1670528650283813,
        -1.3429491519927979,
        -0.35620346665382385,
        0.5061102509498596,
        0.1185610219836235 ] )
wi = np.array([[ 0.4075218737125397,
        0.6494815349578857,
        -1.7012808322906494,
        0.023210864514112473,
        -0.15263228118419647,
        0.2533695697784424
      ],
      [ 0.766633152961731,
        1.5523567199707031,
        -0.4652297794818878,
        1.7160581350326538,
        0.04102218151092529,
        0.7520620226860046 ],
      [  2.1420960426330566,
        0.11224153637886047,
        1.8871101140975952,
        0.8059232831001282,
        1.2740586996078491,
        0.0408564954996109 ]])

bi = np.array([0.2300858050584793,
      0.049704164266586304,
      -0.5237800478935242,
      0.9377124905586243,
      0.5333978533744812,
      -0.1429576575756073])


wo = np.array([  [ 1.4028840065002441 ],
      [  -2.323295831680298 ],
      [ -2.946080207824707 ],
      [  -2.6922714710235596 ],
      [ -2.4723105430603027 ],
      [  -3.3437230587005615 ] ])

bo = [-1.505]

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def linear(x,w,b):
    z = np.matmul(x,w)
    z = z+b
    return z

def fnn(x):
    '''   x = [[1.0,-1.0,-1.0 ]] 1*3 '''
    zi = linear(x, wi, bi)
    ai = sigmoid(zi)
    
    zh = linear(ai, wh, bh)
    ah = sigmoid(zh)

    zo = linear(ah, wo, bo)
    ao = sigmoid(zo)  # 1*1
    return ao

    ''' compute the gradient of a neural network '''
    '''  dfdx  '''

def dfndx(x):
    zi = linear(x, wi, bi)       # 1*3 3*6
    ai = sigmoid(zi)             # 1*6
    
    zh = linear(ai, wh, bh)
    ah = sigmoid(zh)

    zo = linear(ah, wo, bo)
    ao = sigmoid(zo)  # 1*1
    #--------------------------------
    
    sp_i = sigmoid_p(zi)  
    sp_i = np.transpose(sp_i)
    print(sp_i.shape)
    d_i  = sp_i*wi.transpose()     # 1*6 6*3
    #d_i = np.sum(d_i,axis=0)
    print('\n d_i shape: \n',d_i.shape)
    sp_h = sigmoid_p(zh)  
    sp_h = np.transpose(sp_h)
    
    d_h  = np.matmul(sp_h*wh.transpose(),d_i)     # sp*wh^t: 6*1 and 6*3
    #d_h  = np.sum(d_h,axis=0)
    
    sp_o = sigmoid_p(zo)  
    sp_o = np.transpose(sp_o)
    d_o  = np.matmul(sp_o*wo.transpose(),d_h)
    print('\n d_o shape: \n', d_o.shape)
    print('\n d_o: \n', d_o)
    return d_o

x = np.array([[1.0,-1.0,-1.0 ]])
print('\n fnn(x=',x,') \n',fnn(x))

g = egrad(fnn)(x)
print('\n df/dx (autograd): \n',g)

g_ = dfndx(x)
# print(g_)
