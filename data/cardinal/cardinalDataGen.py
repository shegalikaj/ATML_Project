import numpy as np
from scipy.ndimage.interpolation import rotate
import random

import sys
sys.path.append('../')

from utils import *

def cardinalDataGen(seed, angle=0, filename='', numTrainingPoints=20, unseenConcept=''):
    random.seed(seed)

    f = StringFileInterface(filename)

    for i in range(0, numTrainingPoints):
        mat, answer = generateUniqueCardinalWorldAndAnswer(f, angle, unseenConcept, False)
        f.write('\n\nWorld:\n')
        f.write(np.array2string(mat))
        f.write('\nAnswer:')
        f.write(answer)

    mat, answer = generateUniqueCardinalWorldAndAnswer(f, angle, unseenConcept, True)
    f.write('\n\nWorld:\n')
    f.write(np.array2string(mat))
    f.write('\nAnswer:')

    f.close()

    if not(filename):
        return (f.data, answer)

def generateUniqueCardinalWorldAndAnswer(f, angle, unseenConcept='', generateForUnseenConcept=False):
    # Size of the matrix
    # It has to be odd times odd matrix
    m = 2 * random.randint(1, 3) + 1
    n = 2 * random.randint(1, 3) + 1

    # Position of the true value
    do = True
    while do: # Just to ensure that xPos, yPos is not at the middle
        xPos = random.randint(0, m - 1)
        yPos = random.randint(0, n - 1)
        do = ((xPos == m/2 - 0.5) and (yPos == n/2 - 0.5))

    mat = np.zeros([n, m])
    mat[yPos, xPos] = 1

    # Rotate the matrix by the angle provided
    rotatedMat = rotate(mat, angle=angle, reshape=True)
    rotatedMat[abs(rotatedMat) < 0.01] = 0

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

    # Recursively call and return function if:
    # 1. rotatedMat is already present in 'f.data'
    # 2. answer is unseen concept and we are not generating for unseen concept
    # 3. answer is not unseen concept and we are generating for unseen concept
    if ((np.array2string(rotatedMat) in f.data)
            or (answer == unseenConcept and ~generateForUnseenConcept)
            or (answer != unseenConcept and generateForUnseenConcept)):
        return generateUniqueCardinalWorldAndAnswer(f, angle, unseenConcept, generateForUnseenConcept)
    # TODO: This can be done better.
    # Running it recursively until we find something that works for us is inefficient.

    return rotatedMat, answer
