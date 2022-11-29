#!/bin/bash

# first tests, using mbert here (note model_type and model_name_or_path!)
# POS
python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_POS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 \
  --task_name "morpho-class-binary" \
  --output_dir out/POS/xlmr-large/train/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --no_cuda

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_POS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 \
  --task_name "morpho-class-binary" \
  --output_dir out/POS/xlmr-large/traintest/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --eval_set "test" \
  --no_cuda

###############
# MULTIPOS
#
python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS/xlmr-large/train/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --no_cuda

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS/xlmr-large/traintest/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --eval_set "test" \
  --no_cuda


###############
# INDUB
#
python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_INDUB" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/INDUB/xlmr-large/train/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --no_cuda

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_INDUB" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/INDUB/xlmr-large/traintest/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --eval_set "test" \
  --no_cuda

###############
# FEAT

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_FEAT" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-class-binary" \
  --output_dir out/FEAT/xlmr-large/train/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --no_cuda

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_FEAT" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-class-binary" \
  --output_dir out/FEAT/xlmr-large/traintest/ \
  --no_distillation \
  --do_eval \
  --do_train \
  --eval_set "test" \
  --no_cuda
