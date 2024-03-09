# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

# BASE="sarvamai/OpenHathi-7B-Hi-v0.1-Base"
# QUANTS=""
# ADAPTED="ai4bharat/airavata"
# RESULTS="results"

echo "Evaluating BoolQ for each model..."

IFS=','

# ============================================================
#                   Base Models
# ============================================================

for model_path_or_name in $BASE; do
    model_name=$(basename "${model_path_or_name}")

    # -------------------------------------------------------------
    #                            BoolQ
    # -------------------------------------------------------------

    echo "Evaluating $model_name on boolq ..."

    # BoolQ zero-shot
    python3 -m eval.boolq.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/boolq/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8

    # BoolQ 5-shot
    python3 -m eval.boolq.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/boolq/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8
        
    # -------------------------------------------------------------
    #                       Indic BoolQ
    # -------------------------------------------------------------

    echo "Evaluating $model_name on boolq-hi ..."

    # Indic BoolQ zero-shot
    python3 -m eval.boolq.run_translated_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/boolq-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8

    # Indic BoolQ 5-shot
    python3 -m eval.boolq.run_translated_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/boolq-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8

done


# ============================================================
#                   Quantized Models
# ============================================================



for model_path_or_name in $QUANTS; do
    model_name=$(basename "${model_path_or_name}")

    # -------------------------------------------------------------
    #                            BoolQ
    # -------------------------------------------------------------

    echo "Evaluating $model_name on boolq ..."

    # BoolQ zero-shot
    python3 -m eval.boolq.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/boolq/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8

    # BoolQ 5-shot
    python3 -m eval.boolq.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/boolq/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8
        
    # -------------------------------------------------------------
    #                       Indic BoolQ
    # -------------------------------------------------------------

    echo "Evaluating $model_name on boolq-hi ..."

    # Indic BoolQ zero-shot
    python3 -m eval.boolq.run_translated_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/boolq-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8

    # Indic BoolQ 5-shot
    python3 -m eval.boolq.run_translated_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/boolq-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8

done


# ============================================================
#                   Adapted Models
# ============================================================


for model_path_or_name in $ADAPTED; do
    model_name=$(basename "${model_path_or_name}")

    # -------------------------------------------------------------
    #                            BoolQ
    # -------------------------------------------------------------

    echo "Evaluating adapted $model_name on boolq ..."

    # BoolQ zero-shot
    python3 -m eval.boolq.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/boolq/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # BoolQ 5-shot
    python3 -m eval.boolq.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/boolq/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION
        
    # -------------------------------------------------------------
    #                       Indic BoolQ
    # -------------------------------------------------------------

    echo "Evaluating adapted $model_name on boolq-hi ..."

    # Indic BoolQ zero-shot
    python3 -m eval.boolq.run_translated_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/boolq-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # Indic BoolQ 5-shot
    python3 -m eval.boolq.run_translated_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/boolq-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

done
