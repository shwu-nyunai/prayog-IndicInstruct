# BASE="sarvamai/OpenHathi-7B-Hi-v0.1-Base"
# QUANTS=""
# ADAPTED="ai4bharat/airavata"
# RESULTS="results"

# export CUDA_VISIBLE_DEVICES=0

echo "Evaluating Hellaswag for each model..."

IFS=','

# ============================================================
#                   Base Models
# ============================================================

for model_path_or_name in $BASE; do
    model_name=$(basename "${model_path_or_name}")

    # -------------------------------------------------------------
    #                             MMLU
    # -------------------------------------------------------------
    echo "Evaluating $model_name on MMLU ..."
    # MMLU zero-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir data/eval/mmlu \
        --save_dir "$RESULTS/mmlu/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1

    # MMLU 5-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 5 \
        --data_dir data/eval/mmlu \
        --save_dir "$RESULTS/mmlu/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1
    
    # -------------------------------------------------------------
    #                       Indic MMLU
    # -------------------------------------------------------------
    echo "Evaluating $model_name on Indic MMLU ..."
    
    # Indic MMLU zero-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir data/eval/mmlu_hi_translated \
        --save_dir "$RESULTS/mmlu-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1

    # Indic MMLU 5-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 5 \
        --data_dir data/eval/mmlu_hi_translated \
        --save_dir "$RESULTS/mmlu-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1
done


# ============================================================
#                   Quantized Models
# ============================================================

for model_path_or_name in $QUANTS; do
    model_name=$(basename "${model_path_or_name}")

    # -------------------------------------------------------------
    #                             MMLU
    # -------------------------------------------------------------
    echo "Evaluating $model_name on MMLU ..."
    # MMLU zero-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir data/eval/mmlu \
        --save_dir "$RESULTS/mmlu/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 1

    # MMLU 5-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 5 \
        --data_dir data/eval/mmlu \
        --save_dir "$RESULTS/mmlu/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 1
    
    # -------------------------------------------------------------
    #                       Indic MMLU
    # -------------------------------------------------------------
    echo "Evaluating $model_name on Indic MMLU ..."
    
    # Indic MMLU zero-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir data/eval/mmlu_hi_translated \
        --save_dir "$RESULTS/mmlu-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 1

    # Indic MMLU 5-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 5 \
        --data_dir data/eval/mmlu_hi_translated \
        --save_dir "$RESULTS/mmlu-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 1
done

# ============================================================
#                   Adapted Models
# ============================================================

for model_path_or_name in $ADAPTED; do
    model_name=$(basename "${model_path_or_name}")
    # -------------------------------------------------------------
    #                             MMLU
    # -------------------------------------------------------------

    echo "Evaluating $model_name on MMLU ..."

    # MMLU zero-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir data/eval/mmlu \
        --save_dir "$RESULTS/mmlu/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # MMLU 5-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 5 \
        --data_dir data/eval/mmlu \
        --save_dir "$RESULTS/mmlu/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # -------------------------------------------------------------
    #                       Indic MMLU
    # -------------------------------------------------------------

    echo "Evaluating $model_name on Indic MMLU ..."

    # Indic MMLU zero-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir data/eval/mmlu_hi_translated \
        --save_dir "$RESULTS/mmlu-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION


    # Indic MMLU 5-shot
    python3 -m eval.mmlu.run_eval \
        --ntrain 5 \
        --data_dir data/eval/mmlu_hi_translated \
        --save_dir "$RESULTS/mmlu-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION
done