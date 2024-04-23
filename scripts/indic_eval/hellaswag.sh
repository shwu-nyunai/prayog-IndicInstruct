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
    #                       Hellaswag
    # -------------------------------------------------------------


    echo "Evaluating $model_name on Hellaswag ..."

    # Hellaswag zero-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/hellaswag/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1

    # Hellaswag 5-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/hellaswag/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1
    

    # -------------------------------------------------------------
    #                       Indic Hellaswag
    # -------------------------------------------------------------

    echo "Evaluating $model_name on Indic Hellaswag ..."

    # Indic Hellaswag zero-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 0 \
        --dataset "Thanmay/hellaswag-translated" \
        --save_dir "$RESULTS/hellaswag-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1

    # Indic Hellaswag 5-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 5 \
        --dataset "Thanmay/hellaswag-translated" \
        --save_dir "$RESULTS/hellaswag-hi/$model_name-5shot" \
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
    #                       Hellaswag
    # -------------------------------------------------------------


    echo "Evaluating $model_name on Hellaswag ..."

    # Hellaswag zero-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/hellaswag/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 1

    # Hellaswag 5-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/hellaswag/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 1
    

    # -------------------------------------------------------------
    #                       Indic Hellaswag
    # -------------------------------------------------------------

    echo "Evaluating $model_name on Indic Hellaswag ..."

    # Indic Hellaswag zero-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 0 \
        --dataset "Thanmay/hellaswag-translated" \
        --save_dir "$RESULTS/hellaswag-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 1

    # Indic Hellaswag 5-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 5 \
        --dataset "Thanmay/hellaswag-translated" \
        --save_dir "$RESULTS/hellaswag-hi/$model_name-5shot" \
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
    #                       Hellaswag
    # -------------------------------------------------------------
    echo "evaluating $model_name on Hellaswag ..."
    # Hellaswag zero-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 0 \
        --save_dir "$RESULTS/hellaswag/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION


    # Hellaswag 5-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 5 \
        --save_dir "$RESULTS/hellaswag/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # -------------------------------------------------------------
    #                       Indic Hellaswag
    # -------------------------------------------------------------
    echo "evaluating $model_name on Indic Hellaswag ..."

    # Indic Hellaswag zero-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 0 \
        --dataset "Thanmay/hellaswag-translated" \
        --save_dir "$RESULTS/hellaswag-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION


    # Indic Hellaswag 5-shot
    python3 -m eval.hellaswag.run_eval \
        --ntrain 5 \
        --dataset "Thanmay/hellaswag-translated" \
        --save_dir "$RESULTS/hellaswag-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION
done