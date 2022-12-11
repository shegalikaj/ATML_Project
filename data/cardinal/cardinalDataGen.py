import numpy as np
from scipy.ndimage.interpolation import rotate
import random

import sys
sys.path.append('../')

from utils import *

# TODO: Confirm if the rotation implementation is consistent with the paper
def cardinalDataGen(seed, angle=0, filename=''):
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

        # Rotate the matrix by the angle provided
        rotatedMat = rotate(mat, angle=angle, reshape=True)
        rotatedMat[abs(rotatedMat) < 0.01] = 0

        # Prompt
        f.write('\n\nWorld:\n')
        f.write(np.array2string(rotatedMat))
        f.write('\nAnswer:')
        if xPos > m/2 - 0.5:
            if yPos > n/2 - 0.5:
                answer = 'southeast'
            elif yPos < n/2 - 0.5:
                answer = 'northeast'
            else:
                answer = 'east'
        elif xPos < m/2 - 0.5:
            if yPos > n/2 - 0.5:
                answer = 'southwest'
            elif yPos < n/2 - 0.5:
                answer = 'northwest'
            else:
                answer = 'west'
        else:
            if yPos > n/2 - 0.5:
                answer = 'south'
            elif yPos < n/2 - 0.5:
                answer = 'north'
            else:
                answer = 'middle' # This shouldn't happen

        if i < 20 - 1:
            f.write(answer)

    f.close()

    if not(filename):
        # TODO: Extract expected answer and prompt from f.data
        return (f.data, answer)
