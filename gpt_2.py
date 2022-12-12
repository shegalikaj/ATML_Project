from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
 
promt='''
World:
[[0. 0. 0.]
 [0. 0. 0.]
 [1. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
Answer:left

World:
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0.]]
Answer:
'''

promt = promt.strip()
 
input_ids = tokenizer.encode(
    promt,
    add_special_tokens=False,
    return_tensors="pt",
    add_space_before_punct_symbol=True
)
 
output_ids = model.generate(
    input_ids=input_ids,
    do_sample=True,
    # max_length=10,  # desired output sentence length
    pad_token_id=model.config.eos_token_id,
    max_new_tokens=1
)[0].tolist()
 
generated_text = tokenizer.decode(
    output_ids,
    clean_up_tokenization_spaces=True)
 
print(generated_text)