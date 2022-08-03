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
  --output_dir out/POS/xlmr-large/evaldev/ \
  --no_distillation \
  --do_eval \
  # --do_train

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_POS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 \
  --task_name "morpho-class-binary" \
  --output_dir out/POS/xlmr-large/evaltest/ \
  --no_distillation \
  --do_eval \
  --eval_set "test" \
  # --do_train

###############
# MULTIPOS

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS/xlmr-large/evaldev/ \
  --no_distillation \
  --do_eval \
  # --do_train

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_MULTIPOS" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/MULTIPOS/xlmr-large/evaltest/ \
  --no_distillation \
  --do_eval \
  --eval_set "test" \
  # --do_train


###############
# INDUB

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_INDUB" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/INDUB/xlmr-large/evaldev/ \
  --no_distillation \
  --do_eval \
  # --do_train

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_INDUB" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 \
  --task_name "morpho-class-binary" \
  --output_dir out/INDUB/xlmr-large/evaltest/ \
  --no_distillation \
  --do_eval \
  --eval_set "test" \
  # --do_train


###############
# FEAT

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_FEAT" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-class-binary" \
  --output_dir out/FEAT/xlmr-large/evaldev/ \
  --no_distillation \
  --do_eval \
  # --do_train

python3 cli.py \
  --method pet \
  --data_dir "../testset1/subtask_FEAT" \
  --model_type "xlm-roberta" \
  --model_name_or_path "xlm-roberta-large" \
  --pattern_ids 0 1 2 3 \
  --task_name "morpho-class-binary" \
  --output_dir out/FEAT/xlmr-large/evaltest/ \
  --no_distillation \
  --do_eval \
  --eval_set "test" \
  # --do_train
