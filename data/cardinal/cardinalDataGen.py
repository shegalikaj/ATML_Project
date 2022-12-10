import numpy as np
import random

import sys
sys.path.append('../')

from utils import *

def cardinalDataGen(seed, filename=''):
    random.seed(seed)

    f = StringFileInterface(filename)

    for i in range(0, 20):
        # Size of the matrix
        # It has to be odd times odd matrix
        m = 2 * random.randint(1, 3) + 1
        n = 2 * random.randint(1, 3) + 1

        # Position of the true value
        do = True
        while do: # Just to ensure that xPos, yPos is not at the middle
            xPos = random.randint(0, m - 1)
            yPos = random.randint(0, n - 1)
            do = ((xPos == m/2) and (yPos == n/2))

        mat = np.zeros([n, m])
        mat[yPos, xPos] = 1

        # Prompt
        f.write('World:\n')
        f.write(np.array2string(mat))
        if xPos > m/2 - 0.5:
            if yPos > n/2 - 0.5:
                f.write('\nAnswer: southeast\n\n')
            elif yPos < n/2 - 0.5:
                f.write('\nAnswer: northeast\n\n')
            else:
                f.write('\nAnswer: east\n\n')
        elif xPos < m/2 - 0.5:
            if yPos > n/2 - 0.5:
                f.write('\nAnswer: southwest\n\n')
            elif yPos < n/2 - 0.5:
                f.write('\nAnswer: northwest\n\n')
            else:
                f.write('\nAnswer: west\n\n')
        else:
            if yPos > n/2 - 0.5:
                f.write('\nAnswer: south\n\n')
            elif yPos < n/2 - 0.5:
                f.write('\nAnswer: north\n\n')
            else:
                f.write('\nAnswer: middle\n\n') # This shouldn't happen

    f.close()

    if not(filename):
        return f.data
