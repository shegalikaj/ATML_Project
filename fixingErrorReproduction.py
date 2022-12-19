import numpy as np
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    set_seed
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_to_eval = ("gpt2",0)

# The following prompt should work fine
#prompt = "\n\nWorld:\n[[ 0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.   -0.04  0.6   0.36  0.  ]\n [ 0.    0.02 -0.08  0.12  0.01  0.  ]\n [ 0.    0.    0.    0.    0.    0.  ]]\nAnswer:west\n\nWorld:\n[[ 0.    0.    0.    0.    0.    0.  ]\n [ 0.   -0.03  0.    0.    0.    0.  ]\n [ 0.   -0.01  0.01  0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.  ]]\nAnswer:southeast\n\nWorld:\n[[ 0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.02  0.    0.    0.    0.  ]\n [ 0.    0.   -0.04 -0.02  0.01  0.    0.  ]\n [ 0.    0.    0.07  0.31 -0.12  0.03  0.  ]\n [ 0.    0.    0.17  0.6  -0.02  0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.  ]]\nAnswer:"

# The following prompt gives an error
prompt = "\n\nWorld:\n[[ 0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.   -0.04  0.6   0.36  0.  ]\n [ 0.    0.02 -0.08  0.12  0.01  0.  ]\n [ 0.    0.    0.    0.    0.    0.  ]]\nAnswer:west\n\nWorld:\n[[ 0.    0.    0.    0.    0.    0.  ]\n [ 0.   -0.03  0.    0.    0.    0.  ]\n [ 0.   -0.01  0.01  0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.  ]]\nAnswer:southeast\n\nWorld:\n[[ 0.    0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.   -0.09 -0.02  0.  ]\n [ 0.    0.    0.    0.   -0.02  0.29  0.58  0.  ]\n [ 0.    0.    0.    0.02  0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.    0.  ]]\nAnswer:northwest\n\nWorld:\n[[ 0.    0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.15 -0.02  0.    0.  ]\n [ 0.    0.    0.01 -0.09  0.87  0.17 -0.01  0.  ]\n [ 0.   -0.01  0.02 -0.02 -0.06 -0.1   0.01  0.  ]\n [ 0.    0.    0.    0.    0.    0.03  0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.    0.  ]]\nAnswer:southwest\n\nWorld:\n[[ 0.    0.    0.    0.    0.  ]\n [ 0.   -0.05 -0.01  0.    0.  ]\n [ 0.    0.14  0.38  0.    0.  ]\n [ 0.    0.1   0.53  0.    0.  ]\n [ 0.    0.   -0.12 -0.01  0.  ]\n [ 0.    0.    0.02  0.01  0.  ]\n [ 0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.  ]]\nAnswer:south\n\nWorld:\n[[ 0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.  ]\n [ 0.    0.    0.02  0.    0.    0.    0.  ]\n [ 0.    0.   -0.04 -0.02  0.01  0.    0.  ]\n [ 0.    0.    0.07  0.31 -0.12  0.03  0.  ]\n [ 0.    0.    0.17  0.6  -0.02  0.    0.  ]\n [ 0.    0.    0.    0.    0.    0.    0.  ]]\nAnswer:"

tokenizer = GPT2Tokenizer.from_pretrained(model_to_eval[0])

encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt', max_length=4096)

model = GPT2LMHeadModel.from_pretrained(model_to_eval[0]).to(device)

output_ids1 = []
output_ids2 = []
output_ids3 = []
for i in range(0, encoded_input['input_ids'].size(1), 1024):
    input_ids = encoded_input['input_ids'][:, i:i+1024]
    attention_mask = encoded_input['attention_mask'][:, i:i+1024]
    batch_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        #max_new_tokens=5,
        num_return_sequences=3,
        temperature=1,
        top_p=0.85
    )
    print(batch_output.size)
    output_ids1.extend(batch_output[0].tolist())
    output_ids2.extend(batch_output[1].tolist())
    output_ids3.extend(batch_output[2].tolist())

generated_sequences1 = tokenizer.decode(output_ids1, skip_special_tokens=True)
generated_sequences2 = tokenizer.decode(output_ids2, skip_special_tokens=True)
generated_sequences3 = tokenizer.decode(output_ids3, skip_special_tokens=True)


#print(generated_sequences[0].replace(prompt, ""))
#print(generated_sequences[1].replace(prompt, ""))
#print(generated_sequences[2].replace(prompt, ""))

print(generated_sequences1.split('Answer:')[-1])
print(generated_sequences2.split('Answer:')[-1])
print(generated_sequences3.split('Answer:')[-1])
