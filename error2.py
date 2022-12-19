import numpy as np
import time
import os
import pandas as pd
import openai
import random
import re
import csv
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
import sys

openai.api_key = ""

sys.path.append("./data/cardinal")
sys.path.append("./data/spatial")
sys.path.append("./data")

from data.cardinal.cardinalDataGen import cardinalDataGen, cardinalSubspaceDataGen
from data.spatial.spatialDataGen import spatialDataGen
from data.colours.colorDataGen import (
    sub_space_color_generation,
    random_split_color_generation,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


positions = [
    [("left", "right"), "horizontal"],
    [("up", "down"), "vertical"],
    [("top", "bottom"), "vertical"],
]
experiment = 23
num_prompts = 5

pos = positions[1]

(prompt, expectedAnswer) = spatialDataGen(
    experiment,
    angle=0,
    filename="",
    numTrainingPoints=num_prompts,
    unseenConcept="",
    answerValues=pos[0],
    direction=pos[1],
)

print(prompt)
print(expectedAnswer)
