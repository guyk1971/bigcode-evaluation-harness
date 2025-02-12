#!/bin/bash

accelerate launch main.py \
--model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct  \
--tasks humanevalsynthesize-cpp \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt deepseek \
--save_generations_path outputs/generations_humanevalsynthesize-cpp_deepseekv2LiteInst.json \
--metric_output_path outputs/evaluation_humanevalsynthesize-cpp_deepseekv2LiteInst.json \
--max_length_generation 4096 \
--precision bf16

# v0
accelerate launch main.py \
--model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B/snapshots/0396a76181e127dfc13e5c5ec48a8cee09938b02/  \
--tasks humanevalsynthesize-cpp \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt codeqwen \
--save_generations_path outputs/generations_codeqwen25.json \
--metric_output_path outputs/evaluation_codeqwen25.json \
--max_length_generation 4096 \
--precision bf16




# v1
accelerate launch main.py \
--model ~/scratch/models/llm4cuda/qwen2.5coder/7B_Base_v0.1/  \
--tasks humanevalsynthesize-cpp \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt codeqwen \
--save_generations_path outputs/generations_codeqwen25.json \
--metric_output_path outputs/evaluation_codeqwen25.json \
--max_length_generation 4096 \
--precision bf16






accelerate launch main.py \
--model microsoft/phi-2  \
--tasks humanevalsynthesize-cpp \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt continue \
--save_generations_path outputs/generations_phi2.json \
--metric_output_path outputs/evaluation_phi2.json \
--save_every_k_tasks 10 \
--max_length_generation 4096 \
--precision bf16