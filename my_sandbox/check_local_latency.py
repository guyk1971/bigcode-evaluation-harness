import torch
import numpy as np
import argparse
from transformers import AutoModel, AutoTokenizer  # Assuming HuggingFace Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load models
# DSC
# model_a = AutoModel.from_pretrained('DeepSeekCoder-v2-lite-base')
# or 
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)


# Qwen
# model_b = AutoModel.from_pretrained('Qwen/Qwen2.5-Coder-7B')
# or
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B")





# Function to measure latency
def measure_latency(model, tokenizer, context_length,max_new_tokens):
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, context_length)).to(model.device)  # Adjust token range as needed
    dummy_input_text = "This is a test input." * (context_length // len("This is a test input."))
    dummy_input = tokenizer(dummy_input_text, return_tensors="pt").to(model.device)
    # Warm-up GPU
    for _ in range(10):
        _ = model.generate(**dummy_input,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id)

    # Measure inference time
    repetitions = 100
    timings = np.zeros((repetitions,))
    
    with torch.no_grad():
        for i in range(repetitions):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model.generate(**dummy_input,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id)
            end_event.record()

            # Wait for GPU synchronization
            torch.cuda.synchronize()

            # Calculate elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            timings[i] = elapsed_time_ms

    mean_time = np.mean(timings)
    std_time = np.std(timings)
    return mean_time, std_time



def measure_model_latency(model_name):
    # model = AutoModel.from_pretrained(model_name,torch_dtype=torch.bfloat16,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,trust_remote_code=True)

    # Set pad_token_id to eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'running on {device}')
    model.to(device)
    # Define context lengths to test
    context_lengths = [128, 256, 512, 1024]
    max_new_tokens_list = [1,50,100,200]
    # Measure and print results
    for length in context_lengths:
        for max_new_tokens in max_new_tokens_list:
            mean_a, std_a = measure_latency(model,tokenizer, length,max_new_tokens)
            print(f'Context Length: {length}, max new tokens: {max_new_tokens}')
            print(f'{model_name} latency: Mean={mean_a:.2f}ms, Std={std_a:.2f}ms')

    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check model latency")
    parser.add_argument('-m','--model', type=str, default='deepseek-ai/DeepSeek-Coder-V2-Lite-Base', help="model name (default: DSC-v2-lite-base)")
    args = parser.parse_args()
    print(f"measuring inference latency of {args.model}")
    measure_model_latency(args.model)


