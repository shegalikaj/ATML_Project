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

input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt",
    max_length=3072
).to(device)

model = GPT2LMHeadModel.from_pretrained(model_to_eval[0]).to(device)
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

generated_sequences = [
    tokenizer.decode(s.tolist(), skip_special_tokens=True) for s in output_ids
]

print(generated_sequences[0].replace(prompt, ""))
print(generated_sequences[1].replace(prompt, ""))
print(generated_sequences[2].replace(prompt, ""))
