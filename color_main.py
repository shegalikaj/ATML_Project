import openai
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# fields = ['color', 'hexadecimal','R', 'G', 'B']

# df = pd.read_csv('data/colours/extracted_colors.csv', usecols = fields, low_memory = True)
# df['RGB'] = list(zip(df.R, df.G, df.B))

# openai.api_key = "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"

# X = df.R
# Y = df.G
# Z = df.B
# The representation of RGB colors as points in 3D, where the color of each point corresponds to RGB values. 
# X =list(df.R)
# Y =list(df.G)
# Z =list(df.B)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')


# colors = list(df.hexadecimal)
# for i in range(len(X)):
#     ax.scatter(X[i], Y[i], Z[i], color=colors[i])
# plt.show()


# # Color generation 
# def color_generation():
#     color_keys = [i for i in range(len(df.color))]
#     samples = []
#     key_of_the_last_value = None
#     for key, value in enumerate(random.choices(color_keys, k=60)): 
#         if key < 59: 
#             samples.append("RGB: {} Answer: {}".format(df.RGB[value], df.color[value]))
#     else: 
#         key_of_the_last_value = value
#         samples.append("RGB: {} Answer: ".format(df.RGB[value]))

#     return samples, key_of_the_last_value

 
# samples,key_of_the_last_value = color_generation()

# the_last_value = samples[-1]
# gpt_prompt = " ".join(samples)
# gpt_prompt = gpt_prompt.strip()


# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt=gpt_prompt,
#   temperature=1,
#   max_tokens=1,
#   top_p=1,
#   frequency_penalty=0.0,
#   presence_penalty=0.0,
#   logprobs=3
# )

# color = df.RGB[key_of_the_last_value]

# # Use ANSI escape codes to set the text color to the desired RGB values
# print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m")
# print("\u2588" * 10, the_last_value + response['choices'][0]['text'])


# Writing the function color_generation
def color_generation(seed, df):
    random.seed(seed)
    color_keys = [i for i in range(len(df.color))]
    samples = []
    key_of_the_last_value = None
    for key, value in enumerate(random.sample(color_keys, k=60)): 
        if key < 59: 
            samples.append("RGB: {} Answer: {}".format(df.RGB[value], df.color[value]))
    else: 
        key_of_the_last_value = value
        samples.append("RGB: {} Answer: ".format(df.RGB[value]))

    gpt_prompt = " ".join(samples)
    gpt_prompt = gpt_prompt.strip()
    
    return gpt_prompt, samples[-1], key_of_the_last_value

 


