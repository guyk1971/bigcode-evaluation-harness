#!/bin/bash

#################
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
--save_generations_path outputs/generations_codeqwen25_hep-cpp_v0.json \
--metric_output_path outputs/evaluation_codeqwen25_hep-cpp_v0.json \
--max_length_generation 4096 \
--precision bf16


accelerate launch main.py \
--model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B/snapshots/0396a76181e127dfc13e5c5ec48a8cee09938b02/  \
--tasks humanevalsynthesize-py \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt codeqwen \
--save_generations_path outputs/generations_codeqwen25_hep-py_v0.json \
--metric_output_path outputs/evaluation_codeqwen25_hep-py_v0.json \
--max_length_generation 4096 \
--precision bf16


accelerate launch main.py \
--model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B/snapshots/0396a76181e127dfc13e5c5ec48a8cee09938b02/  \
--tasks starcoder_fim \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt codeqwen \
--save_generations_path outputs/generations_codeqwen25_starfim_v0.json \
--metric_output_path outputs/evaluation_codeqwen25_startfim_v0.json \
--max_length_generation 4096 \
--precision bf16



####################
# v1
accelerate launch main.py \
--model ~/models/llm4cuda/qwen2.5coder/7B_Base_v0.1/  \
--tasks humanevalsynthesize-cpp \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt codeqwen \
--save_generations_path outputs/generations_codeqwen25_hep-cpp_v1.json \
--metric_output_path outputs/evaluation_codeqwen25_hep-cpp_v1.json \
--max_length_generation 4096 \
--precision bf16

accelerate launch main.py \
--model ~/models/llm4cuda/qwen2.5coder/7B_Base_v0.1/  \
--tasks humanevalsynthesize-py \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt codeqwen \
--save_generations_path outputs/generations_codeqwen25_hep-py_v1.json \
--metric_output_path outputs/evaluation_codeqwen25_hep-py_v1.json \
--max_length_generation 4096 \
--precision bf16

accelerate launch main.py \
--model ~/models/llm4cuda/qwen2.5coder/7B_Base_v0.1/  \
--tasks starcoder_fim \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt codeqwen \
--save_generations_path outputs/generations_codeqwen25_starfim_v1.json \
--metric_output_path outputs/evaluation_codeqwen25_starfim_v1.json \
--max_length_generation 4096 \
--precision bf16




