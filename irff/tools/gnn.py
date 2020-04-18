#!/usr/bin/env python
from __future__ import print_function



def nnp(nlayer=5,pref='',suf='',dim=4,pd=False,ilib=False,fp=None):
    print('\n# neural network parameters: \n',file=fp)
    if pd:
       pref = "p['" + pref
       suf  = suf+"_'+bd]"
    elif ilib:
       pref = "'" + pref
       suf  = suf+"'"

    for l in range(nlayer):    
        l_      = l+1
        line   = ''
        for i in range(dim):
            i_     = i+1
            if ilib: line += 'line_bond.append(['
            for j in range(dim):
                j_ = j+1
                neuron = pref + 'w' + str(l_) + str(i_) + str(j_) + suf + ','
                line += neuron 
            if ilib: line = line[:-1] + '])'
            line += '\n'
        print(line,file=fp)
   
    for l in range(nlayer):      # b
        l_      = l+1
        line    = 'line_bond.append([' if ilib else ''
        for i in range(dim):
            i_     = i+1
            neuron = pref + 'b' + str(l_) + str(i_) +  suf + ','
            line += neuron
        if ilib: line = line[:-1] + '])'
        print(line,file=fp)

    line = 'line_bond.append([' if ilib else ''                  # o
    for i in range(dim):
        i_     = i+1
        neuron = pref + 'wo' + str(i_) +  suf + ','
        line  += neuron
    line += pref + 'bo' + suf  
    if ilib: line += '])'
    print(line,file=fp)


def generating_nn(nlayer=8,pref='f',dim=8):
    '''  generating neural network codes
       | 11 12 |     | 1 |     | b1 |    | 11X1 + 12X2 + b1 |
       |       |  X  |   |  +  |    | =  |                  |
       | 21 22 |     | 2 |     | b2 |    | 21X1 + 22X2 + b2 |
    '''
    fp = open('nn.py','w')
    # nnp(nlayer=nlayer,pref=pref,dim=dim,fp=fp)
    nnp(nlayer=nlayer,pref=pref,dim=dim,pd=True,fp=fp)
    nnp(nlayer=nlayer,pref=pref,dim=dim,ilib=True,fp=fp)

    # nnp(nlayer=nlayer,pref=pref+'si',dim=dim,fp=fp)
    # nnp(nlayer=nlayer,pref=pref+'si',dim=dim,pd=True,fp=fp)
    # nnp(nlayer=nlayer,pref=pref+'si',dim=dim,ilib=True,fp=fp)

    # nnp(nlayer=nlayer,pref=pref+'pi',dim=dim,fp=fp)
    # nnp(nlayer=nlayer,pref=pref+'pi',dim=dim,pd=True,fp=fp)
    # nnp(nlayer=nlayer,pref=pref+'pi',dim=dim,ilib=True,fp=fp)

    # nnp(nlayer=nlayer,pref=pref+'pp',dim=dim,fp=fp)
    # nnp(nlayer=nlayer,pref=pref+'pp',dim=dim,pd=True,fp=fp)
    # nnp(nlayer=nlayer,pref=pref+'pp',dim=dim,ilib=True,fp=fp)


    print('\n# neural network compute function: \n',file=fp)
    for l in range(nlayer):    
        l_      = l+1
        print('\n',file=fp)
        if l>0: ON_ = ON
        ON  = []
        for i in range(dim):
            i_ = i + 1
            on = 'o'+str(l_)+str(i_)
            ON.append(on)
            line   = on +' = tf.sigmoid('
            for j in range(dim):
                j_ = j + 1
                wn = "p['"+pref+'w'+str(l_)+str(i_)+str(j_)+"_'+bd]"
                bn = "p['"+pref+'b'+str(l_)+str(i_)+"_'+bd]"
                xn = 'x'+str(j_) if l==0 else ON_[j]
                func_ = wn+'*'+xn

                if j_ == dim:
                   func_ += '+ ' + bn+ ')'
                else:
                   func_ += '+'
                line += func_
            print(line,file=fp)


    print('\n',file=fp)
    line   = 'o_ = tf.sigmoid('               # output layer
    for l in range(dim):    
        l_ = l+1
        wn = "p['"+pref+'wo'+str(l_)+"_'+bd]"
        bn = "p['"+pref+'bo'+str(l_)+"_'+bd]"
        xn = 'o'+str(nlayer)+str(l_) 
        line += wn+'*'+xn
        if l_ == dim:
           line += '+'+"p['"+pref+'bo'+"_'+bd]"+')'  
        else:
           line += '+'

    print(line,file=fp)
    fp.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   generating_nn()

