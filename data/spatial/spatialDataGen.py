import numpy as np
from scipy.ndimage.interpolation import rotate
import random

import sys
sys.path.append('../')

from utils import *

# TODO: Confirm if the rotation implementation is consistent with the paper
def spatialDataGen(seed, angle=0, filename=''):
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

        # Rotate the matrix by the angle provided
        rotatedMat = rotate(mat, angle=angle, reshape=False) # TODO: Should it be true?
        rotatedMat[abs(rotatedMat) < 0.01] = 0

        # Prompt
        f.write('\n\nWorld:\n')
        f.write(np.array2string(rotatedMat))
        f.write('\nAnswer:')
        if xPos > m/2:
            answer = 'right'
        else:
            answer = 'left'

        if i < 20 - 1:
            f.write(answer)

    f.close()

    if not(filename):
        # TODO: Extract expected answer and prompt from f.data
        return (f.data, answer)
