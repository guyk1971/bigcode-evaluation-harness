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


accelerate launch main.py \
--model Qwen/CodeQwen1.5-7B-Chat  \
--tasks humanevalsynthesize-cpp \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt codeqwen \
--save_generations_path outputs/generations_codeqwen15.json \
--metric_output_path outputs/evaluation_codeqwen15.json \
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