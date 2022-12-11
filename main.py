import numpy as np
import time

import openai
openai.api_key = ''

import sys
sys.path.append('./data/cardinal')
sys.path.append('./data/spatial')
sys.path.append('./data')
from cardinalDataGen import cardinalDataGen
from spatialDataGen import spatialDataGen

numModels = 5
numTimesRepeatExperiment = 10

def evaluateInModel(modelNumber, prompt):
    # TODO: include GPT-2 models
    if (modelNumber == 0):
        # Run it on GPT-3
        prompt = prompt.strip()
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=1,
            max_tokens=1,
            top_p=0.85,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=3
        )
        # Requests per minute limit: 60.000000 / min.
        time.sleep(2)
        return response['choices'][0]['text']

    # else:
        # Run it on GPT-2, models with different sizes
    return '???'


# TODO: Experiment runner functions

def experimentWithSpatial():
    print('-' * 10)
    print(f'Experiment with spatial data:')
    statistics = np.zeros([numModels, numTimesRepeatExperiment])

    for seed in range(numTimesRepeatExperiment):
        (prompt, expectedAnswer) = spatialDataGen(seed)

        for i in range(numModels):
            answer = evaluateInModel(i, prompt)
            statistics[i, seed] = (answer == expectedAnswer)
            #print(f'Expected: {expectedAnswer}, Actual: {answer}')

    print(np.array2string(statistics))
    print('-' * 10)

# TODO: Running the experiments

experimentWithSpatial()

#(prompt, expectedAnswer) = spatialDataGen(1)
#print(prompt)
