
# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
# export CUDA_VISIBLE_DEVICES=0

# BASE="sarvamai/OpenHathi-7B-Hi-v0.1-Base"
# QUANTS=""
# ADAPTED="ai4bharat/airavata"
# RESULTS="results"

echo "Evaluating WinoGrande for each model..."

IFS=','

# ============================================================
#                   Base Models
# ============================================================

for model_path_or_name in $BASE; do
    model_name=$(basename "${model_path_or_name}")

    # -------------------------------------------------------------
    #                            WinoGrande
    # -------------------------------------------------------------

    echo "Evaluating $model_name on winogrande ..."

    # WinoGrande zero-shot
    python3 -m eval.winogrande.run_eval \
        --save_dir "$RESULTS/winogrande/$model_name-0shot" \
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
    #                            WinoGrande
    # -------------------------------------------------------------

    echo "Evaluating $model_name on winogrande ..."

    # WinoGrande zero-shot
    python3 -m eval.winogrande.run_eval \
        --save_dir "$RESULTS/winogrande/$model_name-0shot" \
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
    #                            WinoGrande
    # -------------------------------------------------------------

    echo "Evaluating adapted $model_name on winogrande ..."

    # WinoGrande zero-shot
    python3 -m eval.winogrande.run_eval \
        --save_dir "$RESULTS/winogrande/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

done
