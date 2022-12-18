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

def cardinalDataGen(seed, angle=0, filename='', numTrainingPoints=20, unseenConcept=''):
    random.seed(seed)

    f = StringFileInterface(filename)

    for i in range(0, numTrainingPoints):
        answer, mat = generateUniqueCardinalWorldAndAnswer(f, angle, unseenConcept, False)
        f.write('\n\nWorld:\n')
        f.write(np.array2string(mat))
        f.write('\nAnswer:')
        f.write(answer)

    answer, mat = generateUniqueCardinalWorldAndAnswer(f, angle, unseenConcept, True)
    f.write('\n\nWorld:\n')
    f.write(np.array2string(mat))
    f.write('\nAnswer:')

    f.close()

    if not(filename):
        return (f.data.strip(), answer)

def generateUniqueCardinalWorldAndAnswer(f, angle, unseenConcept='', generateForUnseenConcept=False):
    if unseenConcept == '':
        generateForUnseenConcept = False

    if (generateForUnseenConcept):
        answer = unseenConcept
    else:
        space = {
            'north', 'east', 'west', 'south',
            'southeast', 'northeast', 'southwest', 'northwest'
        }
        answer = random.choice(tuple(space - {unseenConcept}))

    rotatedMat = cardinalGenPointOfType(answer, angle)

    # Recursively call and return function if:
    # rotatedMat is already present in 'f.data'
    if (np.array2string(rotatedMat) in f.data):
        return generateUniqueCardinalWorldAndAnswer(f, angle, unseenConcept, generateForUnseenConcept)

    return answer, rotatedMat


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
        xPos = random.randint(math.ceil(m/2), m - 1)
        yPos = random.randint(math.ceil(n/2), n - 1)
    elif type == 'northeast':
        # xPos > m/2 - 0.5:
        # yPos < n/2 - 0.5:
        xPos = random.randint(math.ceil(m/2), m - 1)
        yPos = random.randint(0, math.floor(n/2) - 1)
        print('x')
    elif type == 'east':
        # xPos > m/2 - 0.5:
        # yPos = n/2 - 0.5
        xPos = random.randint(math.ceil(m/2), m - 1)
        # 'n' has to have an exact middle row
        if n % 2 == 0:
            n = n + 1
        yPos = n/2 - 0.5
    elif type == 'southwest':
        # xPos < m/2 - 0.5:
        # yPos > n/2 - 0.5:
        xPos = random.randint(0, math.floor(m/2) - 1)
        yPos = random.randint(math.ceil(n/2), n - 1)
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
        yPos = random.randint(math.ceil(n/2), n - 1)

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

    xPos = int(xPos)
    yPos = int(yPos)

    mat = np.zeros([n, m])
    mat[yPos, xPos] = 1

    # Rotate the matrix by the angle provided
    rotatedMat = rotate(mat, angle=angle, reshape=True)
    rotatedMat = rotatedMat.round(2)
    rotatedMat[rotatedMat == 0] = 0

    return rotatedMat


# For debugging
#experiment = 2
#num_prompts = 5
#x,y = cardinalDataGen(experiment, angle=0, filename='', numTrainingPoints=num_prompts, unseenConcept='')
#print(x)
#print(y)
