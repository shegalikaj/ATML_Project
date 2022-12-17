import numpy as np
from scipy.ndimage.interpolation import rotate
import random

import sys
sys.path.append('../')

from utils import *

# TODO: GENERALISATION TO UNSEEN CONCEPTS -


# This is for 'GENERALISATION TO UNSEEN WORLDS'
def spatialDataGen(seed, angle=0, filename='', numTrainingPoints=20):
    random.seed(seed)

    f = StringFileInterface(filename)

    for i in range(0, numTrainingPoints + 1):
        rotatedMat, answer = generateUniqueWorldAndAnswer(f, angle)

        # Prompt
        f.write('\n\nWorld:\n')
        f.write(np.array2string(rotatedMat))
        f.write('\nAnswer:')

        if i < numTrainingPoints:
            f.write(answer)

    f.close()

    if not(filename):
        # TODO: Extract expected answer and prompt from f.data
        return (f.data, answer)

def generateUniqueWorldAndAnswer(f, angle):
    # Size of the matrix
    m = random.randint(2, 7)
    n = random.randint(1, 7)

    # Position of the true value
    do = True
    while do: # Just to ensure that xPos is not at the middle
        xPos = random.randint(0, m - 1)
        do = (xPos == m/2 - 0.5)
    yPos = random.randint(0, n - 1)

    mat = np.zeros([n, m])
    mat[yPos, xPos] = 1

    # Rotate the matrix by the angle provided
    rotatedMat = rotate(mat, angle=angle, reshape=False) # TODO: Should it be true?
    rotatedMat[abs(rotatedMat) < 0.01] = 0

    if xPos > m/2:
        answer = 'right'
    else:
        answer = 'left'

    # Confirm if the rotatedMat is already present in 'f';
    # If it is, recursively call and return this function
    ## Is np.array2string(rotatedMat) present in f.data ?
    if (np.array2string(rotatedMat) in f.data):
        return generateUniqueWorldAndAnswer(f, angle)

    return rotatedMat, answer
