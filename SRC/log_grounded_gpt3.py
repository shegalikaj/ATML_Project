import requests
from create_grounded_samples import *
import sys
import json
import os
import operator
import numpy as np

# gpt3 import
import openai

# python gpt-2/log_grounded_gpt.py gpt3 train 0 3p-3s


random.seed(30)

def get_sys_args():
    model_idx, split, start_idx, data = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    return model_idx, split, start_idx, data

def get_spatial_prompts(split="train", model="gpt"):
    if model == "gpt":
        max_len = 1024
    else:
        max_len = 400
    prompt = spatial_prompts(max_len=max_len)
    name = "spatial-0"
    return prompt, name


def get_grounded_words(data="gw"):

    print(data)
    if data == "gw":
        dirpath = "../data/grounded-words"
        files = os.listdir(dirpath)

    elif data == "only-rgb":
        dirpath = "../data/grounded-prompts/"
        files = ["colour-rgbs.tsv"]
    elif data == "only-name":
        dirpath = "../data/grounded-prompts/"
        files = [f for f in os.listdir(dirpath) if "colour-names" in f]
    elif data == "only-hex":
        dirpath = "../data/grounded-prompts/"
        files = [f for f in os.listdir(dirpath) if "colour-hexes" in f]

    print(files)
    words = []
    for file in files:
        print(file)
        f = open(os.path.join(dirpath, file), "r")
        lines = [line.strip() for line in f.readlines()]
        words.append(lines)

    return files, words

def get_dirpaths(type, model_idx, counter, rotation, engine="davinci"):
    dirpath = "../data/grounded-prompts/"
    if rotation is None:
        if type.startswith("ns") or type.startswith("lr") or type.startswith("tb") or type.startswith("ud") or "l-r" in type or "r-l" in type or "u-d" in type or "d-u" in type or "t-b" in type or "b-t" in type:
            prompt_dirpath = dirpath + "spatial-prompts/%s/prompts/%s" % (type, str(counter))
        else:
            prompt_dirpath = dirpath + "colour-prompts/%s/prompts/%s" % (type, str(counter))
    else:
        if type.startswith("ns") or type.startswith("lr") or type.startswith("tb") or type.startswith("ud") or "l-r" in type or "r-l" in type or "u-d" in type or "d-u" in type or "t-b" in type or "b-t" in type:
            prompt_dirpath = dirpath + "spatial-prompts-r%s/%s/prompts/%s" % (str(rotation), type, str(counter))
        else:
            prompt_dirpath = dirpath + "colour-prompts-r%s/%s/prompts/%s" % (str(rotation), type, str(counter))


    if rotation is None:
        dirpath = "../outputs/"
    else:
        dirpath = "../outputs-r%s/" % rotation

    new_dir = dirpath + type
    if os.path.isdir(new_dir) is False:
        os.mkdir(os.path.join(dirpath, type))

    model_name = model_idx + "-" + engine
    op_dirpath = os.path.join(dirpath, type, model_name)
    if os.path.isdir(op_dirpath) is False:
        os.mkdir(op_dirpath)

    op_dirpath = os.path.join(dirpath, type, model_name, str(counter))
    if os.path.isdir(op_dirpath) is False:
        os.mkdir(op_dirpath)

    return prompt_dirpath, op_dirpath


def log_gpt3(seed=0, type="3p-3s", counter=0, rotation=0):

    engine = "davinci"
    model_idx, split, start_idx, data = get_sys_args()
    print(model_idx, data, counter)
    type = data
    prompt_dirpath, op_dirpath = get_dirpaths(type, model_idx, counter, rotation, engine)

    os.environ['OPENAI_API_KEY'] = None # add key here
    openai.api_key = os.getenv("OPENAI_API_KEY")

    files = os.listdir(prompt_dirpath)
    for file in files[:100]:
        f = open(os.path.join(prompt_dirpath, file), "r")
        lines = f.readlines()
        s = "\n".join(line.strip() for line in lines)


        response = openai.Completion.create(
          engine=engine,
          prompt=s,
          temperature=0.5,
          max_tokens=3,
          top_p=1.0,
          frequency_penalty=0.2,
          presence_penalty=0.0,
          stop=["\n"],
          logprobs=3,
        )

        f = open(os.path.join(op_dirpath, file), "w+")
        if response is not None:

            text = response["choices"][0]["text"]
            logprobs = response["choices"][0]["logprobs"]["top_logprobs"]
            first, second, third = "", "", ""
            for item in logprobs:
                keys = sorted(item.items(), key=operator.itemgetter(1), reverse=True)
                first += keys[0][0].strip() + " "
                second += keys[1][0].strip() + " "
                third += keys[2][0].strip() + " "

        else:
            text, first, second, third = "None", "None", "None", "None"
        f.write(text + "\n" + first + "\n" + second + "\n" + third + "\n")

    return text

if __name__=="__main__":
    model_idx, split, start_idx, data = get_sys_args()

    for counter in range(0, 1):
        if model_idx == "gpt3":
            log_gpt3(counter=counter, rotation=None)
