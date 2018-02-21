'''
Created on Oct 20, 2017

@author: rene
'''

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal, rand

if __name__ == '__main__':
    a = np.linspace(0, 10, 100)
    b = np.exp(-a)
    #plt.plot(a,b)
    #plt.show()
    
    x = normal(size=200)
    plt.hist(x, bins=30)
    plt.show()
    
    #reference matplotlib library documentation for other plotting examples