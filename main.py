import sys
sys.path.append('./data/cardinal')
sys.path.append('./data/spatial')
sys.path.append('./data')
from cardinalDataGen import cardinalDataGen
from spatialDataGen import spatialDataGen

import time

numTimesRepeatExperiment = 10

def evaluateInModel(modelNumber, prompt):
    # TODO: Implement this function
    return 'The answer'


# TODO: Experiment runner functions

def experimentWithSpatial():
    for seed in range(numTimesRepeatExperiment):
        (prompt, expectedAnswer) = spatialDataGen(seed)

        for i in range(0, numModels):
            evaluateInModel(0, prompt)


# TODO: Running the experiments


#(prompt, expectedAnswer) = spatialDataGen(1)
#print(expectedAnswer)
