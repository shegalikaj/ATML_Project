import numpy as np
import time
import os
import pandas as pd
import openai
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
) 
import sys
openai.api_key = "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"

sys.path.append('./data/cardinal')
sys.path.append('./data/spatial')
sys.path.append('./data')

from data.cardinal.cardinalDataGen import cardinalDataGen
from data.spatial.spatialDataGen import spatialDataGen
from data.colours.colorDataGen import color_generation

numModels = 5
numTimesRepeatExperiment = 10
#models = (("gpt2",0),("gpt2-medium",0),("gpt2-large",0),("gpt2-xl",0),("gpt3",1))
models = [("gpt2",0)]
print("Experiments to text")
print(models)

fields = ['color', 'hexadecimal','R', 'G', 'B']

df = pd.read_csv('data/colours/extracted_colors.csv', usecols = fields, low_memory = True)
df['RGB'] = list(zip(df.R, df.G, df.B))


def evaluateInModel(modelNumber,model ,prompt,tokenizer=None):
    # TODO: include GPT-2 models

    if (modelNumber == 1):
        # Run it on GPT-3
        prompt = prompt.strip()
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
            n=3
        )
        # Requests per minute limit: 60.000000 / min.
        time.sleep(2)
        #return response['choices'][0]['text']
        return response['choices'][0]['text']

    elif (modelNumber == 0) :
         # gpt2     # 12-layer, 768-hidden, 12-heads, 117M parameters.
                    # OpenAI GPT-2 English model

        
        prompt = prompt.strip()
        input_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Check the documentation of function generate for any change in attributes. 
        # https://huggingface.co/docs/transformers/main_classes/text_generation
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=True,
            # max_length=10,  # desired output sentence length
            pad_token_id=model.config.eos_token_id,
            max_new_tokens=5,
            top_k=3
        )[0].tolist()

        generated_text = tokenizer.decode(
            output_ids,
            clean_up_tokenization_spaces=True)


        # Run it on GPT-2, models with different sizes
        generated_text=print(generated_text.replace(prompt, ''))
        return generated_text
    
    return "Error"





# #(prompt, expectedAnswer) = spatialDataGen(1)
# #print(prompt)

def run_experiment(numTimesRepeatExperiment,models,type_expermients=["grid"]):
    
    tokenizer=None
    loaded_model=None
    statistics = np.zeros([len(models), numTimesRepeatExperiment])
    seeds=list(range(numTimesRepeatExperiment))

    result_collector = []


    for type_exp in type_expermients:

        for k,model_to_eval in enumerate(models):
            
            print(model_to_eval)
            if model_to_eval[1]==0 : #GPT2 model
                tokenizer = GPT2Tokenizer.from_pretrained(model_to_eval[0])
                model = GPT2LMHeadModel.from_pretrained(model_to_eval[0])

                
            else: #model[1]==1 #GPT3 model
                model = openai.Completion
                pass
            
            
            for experiment in range(numTimesRepeatExperiment):

                if type_exp=="grid":
                    (prompt, expectedAnswer) = spatialDataGen(seeds[experiment])

                elif type_exp=="color":
                    (prompt, s, expectedAnswer) = color_generation(seeds[experiment], df)
                    #print(prompt)
                    #print("=========")
                    expectedAnswer  = df.color[expectedAnswer]
                    #print(expectedAnswer)
                    #print("--------------------")
                    # print(expectedAnswer)
                elif type_exp == "cardinal":
                    (prompt, expectedAnswer) = cardinalDataGen(seeds[experiment])

            #experiment
            print(f"model: {model_to_eval[0]},exp: {k}, type: {type_exp}  ")
            #print(f"prompt: {prompt}")
            answer = evaluateInModel(model_to_eval[1],model ,prompt,tokenizer)



            print(f"{model_to_eval[0]} ans: {answer}, real ans:{expectedAnswer}")


run_experiment(numTimesRepeatExperiment,models,["color"])

            
