import torch
from transformers import  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import transformers 
import os



quantized = True
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=quantized,
    bnb_4bit_compute_dtype=torch.bfloat16 if quantized else None
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    device_map='cuda', 
    quantization_config = quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    device_map='cuda', 
    quantization_config = quantization_config
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

def query_llama_with_template(prompt):
    sequences = pipeline(
        f'{prompt}\n',
        do_sample=True,
        top_k=10,
        # max_new_tokens=256,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        truncation = True,
        max_length=400,
    )
    print(f"Question: {prompt[-1]['content']}\n__________________________")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

query_llama_with_template(messages)