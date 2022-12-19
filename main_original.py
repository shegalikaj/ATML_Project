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

numModels = 5
numTimesRepeatExperiment = 1
# models = (("gpt2",0),("gpt2-medium",0),("gpt2-large",0),("gpt2-xl",0),("gpt3",1))
models = [
    ("gpt2", 0),
    ("gpt2-medium", 0),
    ("gpt2-large", 0),
    ("gpt2-xl", 0),
    ("gpt3", 1),
]
print("Experiments to text")
print(models)

fields = ["color", "hexadecimal", "R", "G", "B"]

df = pd.read_csv("data/colours/extracted_colors.csv", usecols=fields, low_memory=True)
df["RGB"] = list(zip(df.R, df.G, df.B))


def evaluateInModel(
    modelNumber,
    model,
    prompt,
    real_ans,
    tokenizer=None,
):
    if modelNumber == 1:
        # Run it on GPT-3
        response = model.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=1,
            max_tokens=5,
            top_p=0.85,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=3,
            best_of=5,
            n=3,
        )
        # Requests per minute limit: 60.000000 / min.
        time.sleep(2)

        top1, top3 = check_outputs(response, prompt, real_ans, modelNumber)

    elif modelNumber == 0:
        # gpt2     # 12-layer, 768-hidden, 12-heads, 117M parameters.
        # OpenAI GPT-2 English model
        # print(prompt)

        # Generate some text using the model
        # input_ids = torch.tensor(tokenizer.encode(prompt, return_tensors='pt')).to(device)
        # output = model.generate(input_ids, max_length=100, temperature=0.7, top_p=0.9, num_return_sequences=1)
        # output_text = tokenizer.decode(output[0], skip_special_tokens=True)

        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt", max_length=3072
        ).to(device)
        # print(f"len_promt{len(prompt)}")
        # print("input_IDS")

        # Check the documentation of function generate for any change in attributes.
        # https://huggingface.co/docs/transformers/main_classes/text_generation
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=True,  # If False Greedy Decoding
            # max_length=10,  # desired output sentence length
            pad_token_id=model.config.eos_token_id,
            max_new_tokens=5,
            num_return_sequences=3,
            # top_k=3,
            temperature=1,
            top_p=0.85,
        )
        # [0].tolist()

        # generated_text = tokenizer.decode(
        #     output_ids,
        #     clean_up_tokenization_spaces=True)

        # print("_____top3____")
        generated_sequences = [
            tokenizer.decode(s.tolist(), skip_special_tokens=True) for s in output_ids
        ]  # clean_up_tokenization_spaces=True)
        top1, top3 = check_outputs(generated_sequences, prompt, real_ans, modelNumber)

    else:
        raise Exception("Unsupported model number")

    return top1, top3


def check_outputs(response, prompt, exp_ans, mod_num):

    top1 = 0
    top3 = 0

    for i in range(3):
        if mod_num == 1:
            string = response["choices"][i]["text"]
        else:
            string = response[i].replace(prompt, "")

        string = string.replace("_", " ").replace("\n", " ").split(" ")
        # print(string)
        filtered_words = [
            word.strip()
            for word in string
            if word.strip().isalpha() and not re.search("RGB", word)
        ]

        # print(f'filtered_words :{filtered_words}')

        res = [
            "".join(filtered_words[i:j])
            for i in range(len(filtered_words))
            for j in range(i + 1, len(filtered_words) + 1)
        ]

        # index = string.find(substring)
        # if index != -1:
        # print("Substring found at index", index)

        # print(res)
        f = -1
        for substr in res:
            # print(f"looking for {substr} in {exp_ans} ")
            index = exp_ans.replace("_", "").find(substr)
            # print(index)
            if f < index:
                f = index
            if i == 0 and f >= 0:
                # print("FOund on first f: {f}")
                top1 = 1
                top3 = 1
                return top1, top3
            elif i > 0 and f >= 0:
                # print(f"FOund on others f: {f}")
                top3 = 1
                return top1, top3

    return top1, top3


# #(prompt, expectedAnswer) = spatialDataGen(1)
# #print(prompt)


def run_experiment_B1(numTimesRepeatExperiment, models):
    print("Experiment for {numTimesRepeatExperiment} rounds")

    tokenizer = None
    loaded_model = None
    statistics = np.zeros([len(models), numTimesRepeatExperiment])
    seeds = list(range(numTimesRepeatExperiment))
    split = ["random", "subspace"]
    rotation_list = ["None", "90", "Random"]
    colors_prim_sec = [
        "RGB:(255, 0, 0) Answer:red",
        "RGB:(0,255,0) Answer:green",
        "RGB:(0,0,255) Answer:blue",
        "RGB:(255,255,0) Answer:yellow",
        "RGB:(0,255,255) Answer:cyan",
        "RGB:(255,0,255) Answer:magenta",
    ]

    index = []

    for k, model_to_eval in enumerate(models):

        # print(model_to_eval)
        if model_to_eval[1] == 0:  # GPT2 model

            model = GPT2LMHeadModel.from_pretrained(model_to_eval[0]).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(model_to_eval[0])

        else:  # model[1]==1 #GPT3 model
            model = openai.Completion

        for sp in split:

            for rotation in rotation_list:

                for experiment in range(numTimesRepeatExperiment):
                    set_seed(experiment)

                    list_ans = []
                    # print("------------------------------------")
                    # print(f"Color GTUC, {sp} split ,model: {model_to_eval[0]},exp: {k}, rotation:{rotation} ")
                    if sp == "random":
                        if rotation == "None":
                            prompt, s, expectedAnswer = random_split_color_generation(
                                experiment,
                                df,
                                rotation_by_90_degree=False,
                                rotation_random=False,
                            )
                            list_ans.append([prompt, s, df.color[expectedAnswer]])

                        elif rotation == "90":
                            prompt, s, expectedAnswer = random_split_color_generation(
                                experiment,
                                df,
                                rotation_by_90_degree=True,
                                rotation_random=False,
                            )
                            list_ans.append([prompt, s, df.color[expectedAnswer]])

                        elif rotation == "Random":
                            prompt, s, expectedAnswer = random_split_color_generation(
                                experiment,
                                df,
                                rotation_by_90_degree=False,
                                rotation_random=True,
                            )
                            list_ans.append([prompt, s, df.color[expectedAnswer]])

                    if sp == "subspace":
                        if rotation == "None":
                            for color_sub in colors_prim_sec:
                                match = re.search(r"\((\d+,\s*\d+,\s*\d+)\)", color_sub)
                                value = match.group(1)
                                parts = value.split(",")
                                rgb_color_sub = tuple(int(x.strip()) for x in parts)
                                prompt, s, expectedAnswer = sub_space_color_generation(
                                    experiment,
                                    df,
                                    rgb_color_sub,
                                    rotation_by_90_degree=False,
                                    rotation_random=False,
                                )
                                list_ans.append([prompt, s, df.color[expectedAnswer]])

                        elif rotation == "90":
                            for color_sub in colors_prim_sec:
                                match = re.search(r"\((\d+,\s*\d+,\s*\d+)\)", color_sub)
                                value = match.group(1)
                                parts = value.split(",")
                                rgb_color_sub = tuple(int(x.strip()) for x in parts)
                                prompt, s, expectedAnswer = sub_space_color_generation(
                                    experiment,
                                    df,
                                    rgb_color_sub,
                                    rotation_by_90_degree=True,
                                    rotation_random=False,
                                )
                                list_ans.append([prompt, s, df.color[expectedAnswer]])

                        elif rotation == "Random":
                            for color_sub in colors_prim_sec:
                                match = re.search(r"\((\d+,\s*\d+,\s*\d+)\)", color_sub)
                                value = match.group(1)
                                parts = value.split(",")
                                rgb_color_sub = tuple(int(x.strip()) for x in parts)
                                prompt, s, expectedAnswer = sub_space_color_generation(
                                    experiment,
                                    df,
                                    rgb_color_sub,
                                    rotation_by_90_degree=False,
                                    rotation_random=True,
                                )
                                list_ans.append([prompt, s, df.color[expectedAnswer]])

                    # experiment
                    # print("+++++++++START++++++++++++++++++++++++")
                    start = time.time()
                    # print(f"Color GTUC, {sp} split ,model: {model_to_eval[0]},exp: {experiment}, rotation:{rotation} ")

                    # print(list_ans)
                    top1_corr = 0
                    top3_corr = 0
                    len_ans = len(list_ans)
                    for ans in list_ans:
                        # print("prompt")
                        # print(ans[0])
                        # print(f"{ans[2]}")
                        # print("")
                        top1_, top3_ = evaluateInModel(
                            model_to_eval[1], model, ans[0], ans[2], tokenizer
                        )
                        top1_corr = top1_corr + top1_
                        top3_corr = top3_corr + top3_
                        end = time.time()
                    # index.append((name, instance.best_sol , opt2_ , s_  , q0_str  ))
                    # print(f"Accuracy top1:{top1_/len_ans}, top2:{top3_/len_ans} time:{end - start}")
                    # print(f"==========time{end - start}=====================")
                    row = [
                        "color",
                        f"{sp} split",
                        model_to_eval[0],
                        rotation,
                        top1_ / len_ans,
                        top3_ / len_ans,
                        end - start,
                    ]
                    print(row)
                    with open(
                        "/content/drive/MyDrive/ATMLData/color_experiment", "a"
                    ) as f:
                        # create the csv writer
                        writer = csv.writer(f)
                        # write a row to the csv file
                        writer.writerow(row)
                        # close the file
                        f.close()


def run_experiment_B2(
    numTimesRepeatExperiment,
    models,
    type_expermients=["grid", "cardinal"],
    num_prompts=20,
):

    tokenizer = None
    loaded_model = None
    statistics = np.zeros([len(models), numTimesRepeatExperiment])
    seeds = list(range(numTimesRepeatExperiment))
    gen_to_unsee = ["world", "concept", "B2.3", "B2.4"]
    rotation_list = ["None", "90", "Random"]
    positions = [
        [("left", "right"), "horizontal"],
        [("up", "down"), "vertical"],
        [("top", "bottom"), "vertical"],
    ]
    cardinal_space = [
        "north",
        "east",
        "west",
        "south",
        "southeast",
        "northeast",
        "southwest",
        "northwest",
    ]

    for k, model_to_eval in enumerate(models):

        print(model_to_eval)
        if model_to_eval[1] == 0:  # GPT2 model
            tokenizer = GPT2Tokenizer.from_pretrained(model_to_eval[0])
            model = GPT2LMHeadModel.from_pretrained(model_to_eval[0])

        else:  # model[1]==1 #GPT3 model
            model = openai.Completion
            pass

        for type_exp in type_expermients:

            for gtu in gen_to_unsee:

                for rotation in rotation_list:

                    for experiment in range(numTimesRepeatExperiment):
                        set_seed(experiment)

                        list_ans = []

                        if type_exp == "grid":
                            if gtu == "world":
                                print(
                                    f"tyep: {type_exp} , {gtu} ,model: {model_to_eval[0]},exp: {experiment}, rotation:{rotation}"
                                )
                                for pos in positions:  # B2.1
                                    if rotation == "None":
                                        # def spatialDataGen(seed, angle=0, filename='', numTrainingPoints=20, unseenConcept='', answerValues=('left', 'right'), direction='horizontal'):

                                        (prompt, expectedAnswer) = spatialDataGen(
                                            experiment,
                                            angle=0,
                                            filename="",
                                            numTrainingPoints=num_prompts,
                                            unseenConcept="",
                                            answerValues=pos[0],
                                            direction=pos[1],
                                        )
                                        # print(len(prompt))
                                        # print(prompt)
                                        # print(expectedAnswer)
                                    elif rotation == "90":
                                        (prompt, expectedAnswer) = spatialDataGen(
                                            experiment,
                                            angle=90,
                                            filename="",
                                            numTrainingPoints=num_prompts,
                                            unseenConcept="",
                                            answerValues=pos[0],
                                            direction=pos[1],
                                        )
                                    elif rotation == "Random":
                                        (prompt, expectedAnswer) = spatialDataGen(
                                            experiment,
                                            angle=random.randint(1, 360),
                                            filename="",
                                            numTrainingPoints=num_prompts,
                                            unseenConcept="",
                                            answerValues=pos[0],
                                            direction=pos[1],
                                        )
                                        # print(len(prompt))
                                        # print(prompt)
                                        # print(expectedAnswer)

                                    list_ans.append([prompt, expectedAnswer])

                            elif gtu == "concept":  # B2.1
                                print(
                                    f"tyep: {type_exp} , {gtu} ,model: {model_to_eval[0]},exp: {experiment}, rotation:{rotation}"
                                )
                                for pos in positions:  # B2.1
                                    for i in range(2):
                                        if rotation == "None":
                                            (prompt, expectedAnswer) = spatialDataGen(
                                                experiment,
                                                angle=0,
                                                filename="",
                                                numTrainingPoints=num_prompts,
                                                unseenConcept=pos[0][i],
                                                answerValues=pos[0],
                                                direction=pos[1],
                                            )
                                        elif rotation == "90":
                                            (prompt, expectedAnswer) = spatialDataGen(
                                                experiment,
                                                angle=90,
                                                filename="",
                                                numTrainingPoints=num_prompts,
                                                unseenConcept=pos[0][i],
                                                answerValues=pos[0],
                                                direction=pos[1],
                                            )
                                        elif rotation == "Random":
                                            (prompt, expectedAnswer) = spatialDataGen(
                                                experiment,
                                                angle=random.randint(1, 360),
                                                filename="",
                                                numTrainingPoints=num_prompts,
                                                unseenConcept=pos[0][i],
                                                answerValues=pos[0],
                                                direction=pos[1],
                                            )

                                        list_ans.append([prompt, expectedAnswer])

                        if type_exp == "cardinal":
                            print(
                                f"tyep: {type_exp} , {gtu} ,model: {model_to_eval[0]},exp: {experiment}, rotation:{rotation}"
                            )
                            if gtu == "world":
                                if rotation == "None":
                                    prompt, expectedAnswer = cardinalDataGen(
                                        experiment,
                                        angle=0,
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        unseenConcept="",
                                    )
                                elif rotation == "90":
                                    prompt, expectedAnswer = cardinalDataGen(
                                        experiment,
                                        angle=90,
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        unseenConcept="",
                                    )
                                elif rotation == "Random":
                                    prompt, expectedAnswer = cardinalDataGen(
                                        experiment,
                                        angle=random.randint(1, 360),
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        unseenConcept="",
                                    )

                                list_ans.append([prompt, expectedAnswer])

                            if gtu == "concept":

                                for i in range(len(cardinal_space)):
                                    print("AQUI ando")

                                    if rotation == "None":
                                        prompt, expectedAnswer = cardinalDataGen(
                                            experiment,
                                            angle=0,
                                            filename="",
                                            numTrainingPoints=num_prompts,
                                            unseenConcept=cardinal_space[i],
                                        )
                                    elif rotation == "90":
                                        prompt, expectedAnswer = cardinalDataGen(
                                            experiment,
                                            angle=90,
                                            filename="",
                                            numTrainingPoints=num_prompts,
                                            unseenConcept=cardinal_space[i],
                                        )
                                    elif rotation == "Random":
                                        prompt, expectedAnswer = cardinalDataGen(
                                            experiment,
                                            angle=random.randint(1, 360),
                                            filename="",
                                            numTrainingPoints=num_prompts,
                                            unseenConcept=cardinal_space[i],
                                        )

                                    list_ans.append([prompt, expectedAnswer])

                            if gtu == "B2.3":
                                if rotation == "None":
                                    prompt, expectedAnswer = cardinalSubspaceDataGen(
                                        experiment,
                                        angle=0,
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        trainSubspace={},
                                    )
                                elif rotation == "90":
                                    prompt, expectedAnswer = cardinalSubspaceDataGen(
                                        experiment,
                                        angle=90,
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        trainSubspace={},
                                    )
                                elif rotation == "Random":
                                    prompt, expectedAnswer = cardinalSubspaceDataGen(
                                        experiment,
                                        angle=random.randint(1, 360),
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        trainSubspace={},
                                    )

                                list_ans.append([prompt, expectedAnswer])

                            if gtu == "B2.4":
                                if rotation == "None":
                                    prompt, expectedAnswer = cardinalSubspaceDataGen(
                                        experiment,
                                        angle=0,
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        trainSubspace={
                                            "north",
                                            "south",
                                            "east",
                                            "west",
                                        },
                                    )
                                elif rotation == "90":
                                    prompt, expectedAnswer = cardinalSubspaceDataGen(
                                        experiment,
                                        angle=90,
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        trainSubspace={
                                            "north",
                                            "south",
                                            "east",
                                            "west",
                                        },
                                    )
                                elif rotation == "Random":
                                    prompt, expectedAnswer = cardinalSubspaceDataGen(
                                        experiment,
                                        angle=random.randint(1, 360),
                                        filename="",
                                        numTrainingPoints=num_prompts,
                                        trainSubspace={
                                            "north",
                                            "south",
                                            "east",
                                            "west",
                                        },
                                    )
                                list_ans.append([prompt, expectedAnswer])

                    # experiment
                    # print("+++++++++START++++++++++++++++++++++++")
                    start = time.time()
                    # print(prompt)
                    # print(expectedAnswer)
                    print(
                        f"tyep: {type_exp} , {gtu} ,model: {model_to_eval[0]},exp: {experiment}, rotation:{rotation}len:{len(prompt)} "
                    )

                    # print(list_ans)
                    if len(list_ans) != 0:

                        top1_ = 0
                        top3_ = 0
                        top1_corr = 0
                        top3_corr = 0
                        len_ans = len(list_ans)
                        for ans in list_ans:
                            # print("prompt")
                            # print(ans[0])
                            # print(f"{ans[2]}")
                            # print("")
                            top1_, top3_ = evaluateInModel(
                                model_to_eval[1], model, ans[0], ans[1], tokenizer
                            )
                            top1_corr = top1_corr + top1_
                            top3_corr = top3_corr + top3_
                            end = time.time()
                        # index.append((name, instance.best_sol , opt2_ , s_  , q0_str  ))
                        # print(f"Accuracy top1:{top1_/len_ans}, top2:{top3_/len_ans} time:{end - start}")
                        # print(f"==========time{end - start}=====================")
                        row = [
                            type_exp,
                            f"GTU {gtu}",
                            model_to_eval[0],
                            rotation,
                            top1_ / len_ans,
                            top3_ / len_ans,
                            end - start,
                        ]
                        print(row)
                        # with open('/content/drive/MyDrive/ATMLData/color_experiment', 'a') as f:#
                        with open("./data/spa_car_exp.csv", "a") as f:
                            # create the csv writer
                            writer = csv.writer(f)
                            # write a row to the csv file
                            writer.writerow(row)
                            # close the file
                            f.close()
                    else:
                        print(
                            f"No test tyep: {type_exp} , {gtu} ,model: {model_to_eval[0]},exp: {experiment}, rotation:{rotation}"
                        )


# run_experiment(numTimesRepeatExperiment,models,["color"])
# run_experiment_B1(numTimesRepeatExperiment,models)
# models =  [("gpt2",0),("gpt2-medium",0),("gpt2-large",0),("gpt2-xl",0),("gpt3",1)]
models = [("gpt2", 0)]
numTimesRepeatExperiment = 1
# spatialDataGen(2, angle=0, filename='', numTrainingPoints=5, unseenConcept='down', answerValues=('up','down'), direction='vertical')
run_experiment_B2(
    numTimesRepeatExperiment,
    models,
    type_expermients=["grid", "cardinal"],
    num_prompts=5,
)