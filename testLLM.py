import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name)
# messages = [
#     {"role": "user", "content": "What is a cat?"}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# inputs = tokenizer(text, return_tensors="pt")
#
# output = llm.generate(
#     **inputs,
#     max_new_tokens=20
# )
#
# print(tokenizer.decode(output[0]))

tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<image>"]}
)
image_token_id = tokenizer.convert_tokens_to_ids("<image>")
print(image_token_id)
messages = [
    {"role": "user", "content": "<image> What is in this image?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
text_embeds = llm.get_input_embeddings()(input_ids)

image_pos = (input_ids[0] == image_token_id).nonzero()[0].item()
print(image_pos)

before = text_embeds[:, :image_pos, :]
after  = text_embeds[:, image_pos + 1:, :]

inputs_embeds = torch.cat([before, after], dim=-1)
print(inputs_embeds.shape)
