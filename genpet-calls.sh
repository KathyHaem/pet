#!/bin/bash

python3 cli.py \
  --method pet \
  --optimizer "adam" \
  --data_dir "../testsets-gen/subtask_OBJ" \
  --output_max_seq_length 32 \
  --model_type "mbart" \
  --model_name_or_path "facebook/mbart-large-cc25" \
  --wrapper_type "generative" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-subj-obj" \
  --output_dir out/OBJ/mbart-large-cc25/testing2/ \
  --do_eval \
  --multi_pattern_training \
  --untrained_model_scoring \
  --no_decoder_prefix \
  --overwrite_output_dir \
  --do_train \
  #--no_cuda \
