# from mpl_toolkits import mplot3d
import pandas as pd
import random
import math
import re


# fields = ['color', 'hexadecimal','R', 'G', 'B']
# df = pd.read_csv('extracted_colors.csv', usecols = fields, low_memory = True)
# df['RGB'] = list(zip(df.R, df.G, df.B))


def random_split_color_generation(seed, df):
    random.seed(seed)
    primary_colors= [ "RGB:(255, 0, 0) Answer:red",  "RGB:(0,255,0) Answer:green", "RGB:(0,0,255) Answer:blue"]
    secondary_colors= [ "RGB:(255,255,0) Answer:yellow",  "RGB:(0,255,255) Answer:cyan", "RGB:(255,0,255) Answer:magenta"]
    samples = []
    color_keys = [i for i in range(len(df.color))]
    key_of_the_last_value = None
    for key, value in enumerate(random.sample(color_keys, k=60)): 
        if key < 64: 
            samples.append("RGB: {} Answer: {}".format(df.RGB[value], df.color[value]))
    else: 
        key_of_the_last_value = value
        samples.append("RGB:{} Answer: ".format(df.RGB[value]))

    samples = primary_colors + secondary_colors + samples
    
    gpt_prompt = " ".join(samples)
    gpt_prompt = gpt_prompt.strip()
    
    print( key_of_the_last_value)
    return gpt_prompt, samples[-1], key_of_the_last_value

def check_euclidian_distance(color_1, color_2, df): 
    distance = math.sqrt((color_2[0]-color_1[0])**2 + (color_2[1]-color_1[1])**2+ (color_2[2]-color_1[2])**2)
    if distance <= 170:
        return True
    else: 
        return False

def sub_space_color_generation(seed, df):
    random.seed(seed)
    primary_secondary_colors = [ "RGB:(255, 0, 0) Answer:red",  "RGB:(0,255,0) Answer:green", "RGB:(0,0,255) Answer:blue", "RGB:(255,255,0) Answer:yellow",  "RGB:(0,255,255) Answer:cyan", "RGB:(255,0,255) Answer:magenta"]
    
    random_color = random.sample(primary_secondary_colors, k=1)[0]
    match = re.search(r"\((\d+,\s*\d+,\s*\d+)\)", random_color)
    value = match.group(1)
    parts = value.split(",")
    rgb_random_color = tuple(int(x.strip()) for x in parts) 
        
    filtered_color_population = list(filter(lambda x: check_euclidian_distance(x, rgb_random_color, df), df.RGB))[:57]
    color_to_be_predict = random.sample(list(set(df.RGB)-set(filtered_color_population)), k=1)[0]
    
    samples = []
    for color in filtered_color_population:
        samples.append("RGB: {} Answer: {}".format(color, df.color[df.index[df['RGB']==color].tolist()[0]]))
    
    samples = primary_secondary_colors+samples + ["RGB: {} Answer: ".format(color_to_be_predict)]
    
    gpt_prompt = " ".join(samples)
    gpt_prompt = gpt_prompt.strip()

    return gpt_prompt, "RGB: {} Answer: ".format(color_to_be_predict), df.index[df['RGB']==color_to_be_predict].tolist()[0]



