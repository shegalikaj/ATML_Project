import numpy as np
import random

random.seed(1)

# Just have left and right here

f = open("spatial.txt", "w")

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
    f.write('World:')
    f.write(np.array2string(mat))
    if xPos > m/2:
        f.write('\nAnswer: right\n\n')
    else:
        f.write('\nAnswer: left\n\n')

f.close()
