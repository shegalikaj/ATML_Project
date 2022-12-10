import numpy as np
import random

import sys
sys.path.append('../')

from utils import *

def spatialDataGen(seed, filename=''):
    random.seed(seed)

    f = StringFileInterface(filename)

    for i in range(0, 20):
        # Size of the matrix
        m = random.randint(2, 4)
        n = random.randint(2, 4)

        # Position of the true value
        do = True
        while do: # Just to ensure that xPos is not at the middle
            xPos = random.randint(0, m - 1)
            do = (xPos == m/2)
        yPos = random.randint(0, n - 1)

        mat = np.zeros([n, m])
        mat[yPos, xPos] = 1

        # Prompt
        f.write('World:\n')
        f.write(np.array2string(mat))
        if xPos > m/2:
            f.write('\nAnswer: right\n\n')
        else:
            f.write('\nAnswer: left\n\n')

    f.close()

    if not(filename):
        return f.data
