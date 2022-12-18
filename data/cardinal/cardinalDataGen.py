import numpy as np
from scipy.ndimage.interpolation import rotate
import random
import math

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
    if unseenConcept == '':
        generateForUnseenConcept = False

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
    rotatedMat = rotatedMat.round(2)
    rotatedMat[rotatedMat == 0] = 0

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
            #answer = 'middle' # This shouldn't happen
            raise Exception('Middle value provided')


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


# Specifically, we show models examples of concepts in one sub-space of
# the world (e.g., north, east, northeast) and then test them on concepts in a different
# sub-space (e.g.,south, west, southwest).
def cardinalSubspaceDataGen(seed, angle=0, filename='', numTrainingPoints=20, trainSubspace={}):
    random.seed(seed)

    f = StringFileInterface(filename)

    space = {
        'north', 'east', 'west', 'south',
        'southeast', 'northeast', 'southwest', 'northwest'
    }

    if len(trainSubspace) == 0:
        # Randomly choose a subspace to train
        trainSubspace = {
            random.choice(tuple(space))
            for i in range(random.randint(0,6))
        }

    # The complementary would be the subspace to test
    testSubspace = space - trainSubspace

    # Sample 'numTrainingPoints' number of datapoints of type 'trainSubspace'
    for i in range(0, numTrainingPoints):
        answer = random.choice(tuple(trainSubspace))
        mat = cardinalGenPointOfType(answer, angle)
        f.write('\n\nWorld:\n')
        f.write(np.array2string(mat))
        f.write('\nAnswer:')
        f.write(answer)

    # Sample one point of type 'testSubspace'
    answer = random.choice(tuple(testSubspace), angle)
    mat = cardinalGenPointOfType(answer)
    f.write('\n\nWorld:\n')
    f.write(np.array2string(mat))
    f.write('\nAnswer:')

    f.close()

    if not(filename):
        return (f.data, answer)

    return mat, answer

def cardinalGenPointOfType(type, angle):
    # Size of the matrix
    # It has to be odd times odd matrix
    m = 2 * random.randint(1, 3) + 1
    n = 2 * random.randint(1, 3) + 1

    # Position of the true value
    if type == 'southeast':
        # xPos > m/2 - 0.5:
        # yPos > n/2 - 0.5:
        xPos = random.randint(math.ceil(m/2), m)
        yPos = random.randint(math.ceil(n/2), n)
    elif type == 'northeast':
        # xPos > m/2 - 0.5:
        # yPos < n/2 - 0.5:
        xPos = random.randint(math.ceil(m/2), m)
        yPos = random.randint(0, math.floor(n/2) - 1)
        print('x')
    elif type == 'east':
        # xPos > m/2 - 0.5:
        # yPos = n/2 - 0.5
        xPos = random.randint(math.ceil(m/2), m)
        # 'n' has to have an exact middle row
        if n % 2 == 0:
            n = n + 1
        yPos = n/2 - 0.5
    elif type == 'southwest':
        # xPos < m/2 - 0.5:
        # yPos > n/2 - 0.5:
        xPos = random.randint(0, math.floor(m/2) - 1)
        yPos = random.randint(math.ceil(n/2), n)
    elif type == 'northwest':
        # xPos < m/2 - 0.5:
        # yPos < n/2 - 0.5:
        xPos = random.randint(0, math.floor(m/2) - 1)
        yPos = random.randint(0, math.floor(n/2) - 1)
    elif type == 'west':
        # xPos < m/2 - 0.5:
        # yPos = n/2 - 0.5:
        xPos = random.randint(0, math.floor(m/2) - 1)
        # 'n' has to have an exact middle row
        if n % 2 == 0:
            n = n + 1
        yPos = n/2 - 0.5
    elif type == 'south':
        # xPos = m/2 - 0.5
        # yPos > n/2 - 0.5
        # 'm' has to have an exact middle row
        if m % 2 == 0:
            m = m + 1
        xPos = m/2 - 0.5
        yPos = random.randint(math.ceil(n/2), n)

    elif type == 'north':
        # xPos = m/2 - 0.5
        # yPos < n/2 - 0.5
        # 'm' has to have an exact middle row
        if m % 2 == 0:
            m = m + 1
        xPos = m/2 - 0.5
        yPos = random.randint(0, math.floor(n/2) - 1)
    else:
        raise Exception('Unsupported direction')

    mat = np.zeros([n, m])
    mat[yPos, xPos] = 1

    # Rotate the matrix by the angle provided
    rotatedMat = rotate(mat, angle=angle, reshape=True)
    rotatedMat = rotatedMat.round(2)
    rotatedMat[rotatedMat == 0] = 0

    return rotatedMat
