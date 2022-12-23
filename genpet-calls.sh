#!/bin/bash

python3 cli.py \
  --method pet \
  --optimizer "adam" \
  --data_dir "../testsets-gen/subtask_SUBJ_de" \
  --output_max_seq_length 16 \
  --model_type "mbart" \
  --model_name_or_path "facebook/mbart-large-cc25" \
  --wrapper_type "generative" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-subj-obj" \
  --output_dir "out/SUBJ_de/mbart-large-cc25/testing" \
  --do_eval \
  --multi_pattern_training \
  --untrained_model_scoring \
  --no_decoder_prefix \
  --do_train \
  --train_examples 1000


# the below will just fail because they've run before so their output directories are not empty
python3 cli.py \
  --method pet \
  --optimizer "adam" \
  --data_dir "../testsets-gen/subtask_OBJ" \
  --output_max_seq_length 8 \
  --model_type "mbart" \
  --model_name_or_path "facebook/mbart-large-cc25" \
  --wrapper_type "generative" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-subj-obj" \
  --output_dir out/OBJ/mbart-large-cc25/testing_less-train/ \
  --do_eval \
  --multi_pattern_training \
  --untrained_model_scoring \
  --no_decoder_prefix \
  --do_train \
  --train_examples 32


python3 cli.py \
  --method pet \
  --optimizer "adam" \
  --data_dir "../testsets-gen/subtask_OBJ_de" \
  --output_max_seq_length 8 \
  --model_type "mbart" \
  --model_name_or_path "facebook/mbart-large-cc25" \
  --wrapper_type "generative" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-subj-obj" \
  --output_dir out/OBJ_de/mbart-large-cc25/testing/ \
  --do_eval \
  --multi_pattern_training \
  --untrained_model_scoring \
  --no_decoder_prefix \
  --do_train \
  --train_examples 100


python3 cli.py \
  --method pet \
  --optimizer "adam" \
  --data_dir "../testsets-gen/subtask_compounds" \
  --output_max_seq_length 16 \
  --model_type "mbart" \
  --model_name_or_path "facebook/mbart-large-cc25" \
  --wrapper_type "generative" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-subj-obj" \
  --output_dir out/compounds/mbart-large-cc25/testing/ \
  --do_eval \
  --multi_pattern_training \
  --untrained_model_scoring \
  --no_decoder_prefix \
  --do_train \


python3 cli.py \
  --method pet \
  --optimizer "adam" \
  --data_dir "../testsets-gen/subtask_compounds" \
  --output_max_seq_length 16 \
  --model_type "mbart" \
  --model_name_or_path "facebook/mbart-large-cc25" \
  --wrapper_type "generative" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-subj-obj" \
  --output_dir out/compounds/mbart-large-cc25/testing-lesstrain/ \
  --do_eval \
  --multi_pattern_training \
  --untrained_model_scoring \
  --no_decoder_prefix \
  --do_train \
  --train_examples 128

