#!/usr/bin/env python
import os
import numpy as np
from ase.structure import molecule
from ase import Atoms
from ase.io import read,write
from amp.descriptor.gaussian import Gaussian
from amp.utilities import hash_images
from ase.visualize import view
from matplotlib import pyplot
from amp.descriptor.gaussian import Gaussian
from amp.utilities import hash_images


images = []
nn1 = read('N2.gen')
# nn1.rotate('y',np.pi/2.0)
# view(nn1)
nn2 = read('N2.gen')
# nn2.positions = nn2.positions 
angs1 = [0,        np.pi/2.0, np.pi/2.0, np.pi/4.0,np.pi/4.0,0.0]
angs2 = [np.pi/2.0,      0.0, np.pi/2.0,-np.pi/4.0,0.0      ,0.0]
for ang1,ang2 in zip(angs1,angs2):
    nn1_ = nn1.copy()
    nn1_.rotate([0,0,1],  ang1)

    nn2_ = nn2.copy()
    nn2_.rotate([0,0,1],  ang2)
    nn2_.positions = nn2_.positions + np.array([3.0,0.0,0.0])

    for atom in nn2_: 
        nn1_.append(atom)
    images.append(nn1_)

view(images)

# # Fingerprint using Amp.
descriptor = Gaussian()
images = hash_images(images, ordered=True)
descriptor.calculate_fingerprints(images)


def barplot(hash, name, title):
    """Makes a barplot of the fingerprint about the O atom."""
    fp = descriptor.fingerprints[hash][0]
    fig, ax = pyplot.subplots()
    ax.bar(range(len(fp[1])), fp[1])
    ax.set_title(title)
    ax.set_ylim(0., 2.)
    ax.set_xlabel('fingerprint')
    ax.set_ylabel('value')
    fig.savefig(name)

for index, hash in enumerate(images.keys()):
    # print(index,hash)
    barplot(hash, 'bplot-%02i.png' % index,
            'different NN positions')


# For fun, make an animated gif.
filenames = ['bplot-%02i.png' % index for index in range(len(images))]
command = ('convert -delay 100 %s -loop 0 animation.gif' %
           ' '.join(filenames))
os.system(command)



