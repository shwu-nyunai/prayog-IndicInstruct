# BASE="sarvamai/OpenHathi-7B-Hi-v0.1-Base"
# QUANTS=""
# ADAPTED="ai4bharat/airavata"
# RESULTS="results"

# -------------------------------------------------------------
# 
# 
#                       IndicXNLI
# 
# 
# -------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0

echo "Evaluating IndicXNLI for each model..."

IFS=','

# ============================================================
#                   Base Models
# ============================================================

for model_path_or_name in $BASE; do
    model_name=$(basename "${model_path_or_name}")

    echo "evaluating $model_name on IndicXNLI ..."

    # zero-shot
    python3 -m eval.indicxnli.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/indicxnli/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8

    # 5-shot
    python3 -m eval.indicxnli.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/indicxnli/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 4
done


# ============================================================
#                   Quantized Models
# ============================================================

for model_path_or_name in $QUANTS; do
    model_name=$(basename "${model_path_or_name}")

    echo "evaluating $model_name on IndicXNLI ..."

    # zero-shot
    python3 -m eval.indicxnli.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/indicxnli/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8

    # 5-shot
    python3 -m eval.indicxnli.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/indicxnli/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 4
done

# ============================================================
#                   Adapted Models
# ============================================================

for model_path_or_name in $ADAPTED; do
    model_name=$(basename "${model_path_or_name}")

    echo "evaluating $model_name on IndicXNLI ..."

    # zero-shot
    python3 -m eval.indicxnli.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/indicxnli/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8 \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


    # 5-shot
    python3 -m eval.indicxnli.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/indicxnli/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

done