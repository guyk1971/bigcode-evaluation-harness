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


