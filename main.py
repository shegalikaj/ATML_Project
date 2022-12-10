import numpy as np

import sys
sys.path.append('./data/cardinal')
sys.path.append('./data/spatial')
sys.path.append('./data')
from cardinalDataGen import cardinalDataGen
from spatialDataGen import spatialDataGen

numModels = 5
numTimesRepeatExperiment = 10

def evaluateInModel(modelNumber, prompt):
    # TODO: Implement this function
    return 'The answer'


# TODO: Experiment runner functions

def experimentWithSpatial():
    statistics = np.zeros([numModels, numTimesRepeatExperiment])

    for seed in range(numTimesRepeatExperiment):
        (prompt, expectedAnswer) = spatialDataGen(seed)

        for i in range(numModels):
            answer = evaluateInModel(0, prompt)
            statistics[i, seed] = (answer == expectedAnswer)

    print(np.array2string(statistics))

# TODO: Running the experiments

experimentWithSpatial()

#(prompt, expectedAnswer) = spatialDataGen(1)
#print(expectedAnswer)
