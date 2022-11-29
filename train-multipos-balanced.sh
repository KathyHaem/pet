#!/bin/bash

# mBERT
#
python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "bert" \
  --model_name_or_path "bert-base-multilingual-cased" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS-EVEN/mbert/train/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --no_cuda \
  --split_examples_evenly

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "bert" \
  --model_name_or_path "bert-base-multilingual-cased" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS-EVEN/mbert/traintest/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --eval_set "test" \
  --no_cuda \
  --split_examples_evenly


###############
# XLM-R Base
#
python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-base" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS-EVEN/xlmr-base/train/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --no_cuda \
  --split_examples_evenly

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-base" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS-EVEN/xlmr-base/traintest/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --eval_set "test" \
  --no_cuda \
  --split_examples_evenly

###############
# XLM-R Large

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS-EVEN/xlmr-large/train/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --no_cuda \
  --split_examples_evenly

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS-EVEN/xlmr-large/traintest/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --eval_set "test" \
  --no_cuda \
  --split_examples_evenly
