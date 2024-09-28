import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Clear out all memory in use by the GPU.
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
# Hello! How can I help you today?
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)

