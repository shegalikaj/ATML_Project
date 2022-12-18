import numpy as np
from scipy.ndimage.interpolation import rotate
import random
import math

class StringFileInterface:
    def __init__(self, filename=''):
        self.toFile = bool(filename)
        if self.toFile:
            self.f = open(filename, "w")
        self.data = ''
    def write(self, str):
        if self.toFile:
            self.f.write(str)
        self.data += str

    def close(self):
        if self.toFile:
            self.f.close()

def spatialDataGen(seed, angle=0, filename='', numTrainingPoints=20, unseenConcept='', answerValues=('left', 'right'), direction='horizontal'):
    random.seed(seed)

    f = StringFileInterface(filename)

    # It is easier to incorporate the direction aspect here rather than
    # changing the entire function with conditional statements
    if direction == 'vertical':
        angle += 90

    for i in range(0, numTrainingPoints):
        mat, answer = generateUniqueSpatialWorldAndAnswer(f, angle, answerValues, unseenConcept, False)
        f.write('\n\nWorld:\n')
        f.write(np.array2string(mat))
        f.write('\nAnswer:'+answer)

    mat, answer = generateUniqueSpatialWorldAndAnswer(f, angle, answerValues, unseenConcept, True)
    f.write('\n\nWorld:\n')
    f.write(np.array2string(mat))
    f.write('\nAnswer:')

    f.close()

    if not(filename):
        return (f.data.strip(), answer)

def generateUniqueSpatialWorldAndAnswer(f, angle, answerValues, unseenConcept='', generateForUnseenConcept=True):
    if unseenConcept == '':
        generateForUnseenConcept = False

    if (generateForUnseenConcept):
        answerIndex = answerValues.index(unseenConcept)
    else:
        space = {0, 1}
        answerIndex = random.choice(tuple(space - {unseenConcept}))

    rotatedMat = spatialGenPointOfType(answerIndex, angle)
    # Recursively call and return function if:
    # rotatedMat is already present in 'f.data'
    if (np.array2string(rotatedMat) in f.data):
        return generateUniqueSpatialWorldAndAnswer(f, angle, unseenConcept, generateForUnseenConcept)

    answer = answerValues[answerIndex]

    return rotatedMat, answer

def spatialGenPointOfType(type, angle):
    # Size of the matrix
    m = random.randint(2, 7)
    n = random.randint(1, 7)

    # Position of the true value
    if type == 1:
        # xPos < m/2 (left)
        xPos = random.randint(0, math.floor(m/2) - 1)
    elif type == 0:
        # xPos > m/2 (right)
        xPos = random.randint(math.ceil(m/2), m - 1)
    else:
        raise Exception('Unsupported direction')
    yPos = random.randint(0, n - 1)

    mat = np.zeros([n, m])
    mat[yPos, xPos] = 1

    # Rotate the matrix by the angle provided
    rotatedMat = rotate(mat, angle=angle, reshape=True)
    rotatedMat = rotatedMat.round(2)
    rotatedMat[rotatedMat == 0] = 0

    return rotatedMat

# For debugging
#x, y = spatialDataGen(2, angle=0, filename='', numTrainingPoints=5, unseenConcept='down', answerValues=('up','down'), direction='vertical')
#print(x)
#print(y)
