# from mpl_toolkits import mplot3d
import pandas as pd
import random
import math
import re


# from mpl_toolkits import mplot3d
import pandas as pd
import random
import math
import re
import numpy as np



# fields = ['color', 'hexadecimal','R', 'G', 'B']
# df = pd.read_csv('extracted_colors.csv', usecols = fields, low_memory = True)
# df['RGB'] = list(zip(df.R, df.G, df.B))

def rotate(point, degree): 
    # Define the rotation matrix
    # rotate in Z
    degree =  np.deg2rad(degree)
    
    rotation_matrix = np.array([[np.cos(degree), -np.sin(degree), 0],
                                [np.sin(degree), np.cos(degree), 0],
                                [0, 0, 1]])

    point = np.array(point)

    rotated_point = np.dot(rotation_matrix, point)

    if degree != 90 :
        rotated_point=rotated_point.round(0).astype(int)

    return tuple(rotated_point)
    

def random_split_color_generation(seed, df,rotation_by_90_degree=False, rotation_random=False):
    random.seed(seed)
    primary_secondary_colors  =[] 
    primary_secondary_rgb = [[[255, 0, 0],  "red"], [[0,255,0], "green"], [[0,0,255], "blue"], [[255,255,0], "yellow"], [[0,255,255], "cyan"], [[255,0,255], "magenta"]]

    if rotation_by_90_degree == False and rotation_random == False : 
        primary_secondary_rgb = [[[255, 0, 0],  "red"], [[0,255,0], "green"], [[0,0,255], "blue"], [[255,255,0], "yellow"], [[0,255,255], "cyan"], [[255,0,255], "magenta"]]
    elif rotation_by_90_degree == True and rotation_random == False:
        for key, value in enumerate(primary_secondary_rgb): 
            primary_secondary_rgb[key][0] = rotate(value[0], 90)
    elif rotation_by_90_degree == False and rotation_random == True:
        for key, value in enumerate(primary_secondary_rgb): 
            
            primary_secondary_rgb[key][0] = rotate(value[0], random.randint(1, 360) )
        
    for key, value in enumerate(primary_secondary_rgb): 
        primary_secondary_colors.append("RGB:{} Answer: {}".format(tuple(value[0]),value[1]))


    samples = []
    color_keys = [i for i in range(len(df.color))]
    key_of_the_last_value = None
    for key, value in enumerate(random.sample(color_keys, k=40)): 
        if key < 40: 
            if rotation_by_90_degree == False and rotation_random == False : 
                # do the rotation by 90 degree
                rgb = df.RGB[value]
            elif rotation_by_90_degree == True and rotation_random == False: 
                # do not rotate
                rgb = rotate(df.RGB[value], 90)
            elif rotation_random == True and rotation_by_90_degree == False:
                # rotate random by a choosen random degree
                rgb = rotate(df.RGB[value], random.randint(1, 360))
            samples.append("RGB: {} Answer: {}".format(rgb, df.color[value]))
    else: 
        key_of_the_last_value = value
        samples.append("RGB:{} Answer: ".format(df.RGB[value]))

    samples = primary_secondary_colors + samples
    
    gpt_prompt = " ".join(samples)
    gpt_prompt = gpt_prompt.strip()
    


    return gpt_prompt, samples[-1], key_of_the_last_value

def check_euclidian_distance(color_1, color_2, df): 
    distance = math.sqrt((color_2[0]-color_1[0])**2 + (color_2[1]-color_1[1])**2+ (color_2[2]-color_1[2])**2)
    if distance <= 170:
        return True
    else: 
        return False

def sub_space_color_generation(seed, df, rgb_random_color, rotation_by_90_degree=False, rotation_random=False,value="RGB:(255, 0, 0) Answer:red"):
    random.seed(seed)

    primary_secondary_colors  =[] 
    primary_secondary_rgb = [[[255, 0, 0],  "red"], [[0,255,0], "green"], [[0,0,255], "blue"], [[255,255,0], "yellow"], [[0,255,255], "cyan"], [[255,0,255], "magenta"]]

    if rotation_by_90_degree == False and rotation_random == False : 
        primary_secondary_rgb = [[[255, 0, 0],  "red"], [[0,255,0], "green"], [[0,0,255], "blue"], [[255,255,0], "yellow"], [[0,255,255], "cyan"], [[255,0,255], "magenta"]]
    elif rotation_by_90_degree == True and rotation_random == False:
        for key, value in enumerate(primary_secondary_rgb): 
            primary_secondary_rgb[key][0] = rotate(value[0], 90)
    elif rotation_by_90_degree == False and rotation_random == True:
        for key, value in enumerate(primary_secondary_rgb): 
            
            primary_secondary_rgb[key][0] = rotate(value[0], random.randint(1, 360) )
        
    for key, value in enumerate(primary_secondary_rgb): 
        primary_secondary_colors.append("RGB:{} Answer: {}".format(tuple(value[0]),value[1]))



    filtered_color_population = list(filter(lambda x: check_euclidian_distance(x, rgb_random_color, df), df.RGB))[:40]
    color_to_be_predict = random.sample(list(set(df.RGB)-set(filtered_color_population)), k=1)[0]
    
    samples = []
    for color in filtered_color_population:
        if rotation_by_90_degree == False and rotation_random == False : 
            # do the rotation by 90 degre
            rgb = df.RGB[df.index[df['RGB']==color].tolist()[0]]
        elif rotation_by_90_degree == True and rotation_random == False: 
            # do not rotate
            rgb = rotate(df.RGB[df.index[df['RGB']==color].tolist()[0]], 90)
            
        elif rotation_random == True and rotation_by_90_degree == False:
            # rotate random by a choosen random degree
            rgb = rotate(df.RGB[df.index[df['RGB']==color].tolist()[0]], random.randint(1, 360))
        #     rgb = rotate(df.RGB[df.color[df.index[df['RGB']==color].tolist()[0]]], random.randint(1, 360))
        samples.append("RGB: {} Answer: {}".format(rgb, df.color[df.index[df['RGB']==color].tolist()[0]]))
    
    samples = primary_secondary_colors+samples + ["RGB: {} Answer: ".format(color_to_be_predict)]
    
    gpt_prompt = " ".join(samples)
    gpt_prompt = gpt_prompt.strip()

    return gpt_prompt, "RGB: {} Answer: ".format(color_to_be_predict), df.index[df['RGB']==color_to_be_predict].tolist()[0]
