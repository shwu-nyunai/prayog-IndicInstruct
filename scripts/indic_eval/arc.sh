# BASE="sarvamai/OpenHathi-7B-Hi-v0.1-Base"
# QUANTS=""
# ADAPTED="ai4bharat/airavata"
# RESULTS="results"

export CUDA_VISIBLE_DEVICES=0

echo "Evaluating ARC-Easy and ARC-Challenge for each model..."

IFS=','

# ============================================================
#                   Base Models
# ============================================================

for model_path_or_name in $BASE; do
    model_name=$(basename "${model_path_or_name}")

    # -------------------------------------------------------------
    #                       ARC-Easy
    # -------------------------------------------------------------

    echo "Evaluating model $model_name on ARC-Easy"

    # ARC-Easy zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai2_arc" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8

    # ARC-Easy 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai2_arc" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8
    
    # -------------------------------------------------------------
    #                       ARC-Challenge
    # -------------------------------------------------------------

    echo "Evaluating model $model_name on ARC-Challenge"

    # ARC-Challenge zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai2_arc" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8

    # ARC-Challenge 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai2_arc" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8
    
    # -------------------------------------------------------------
    #                       Indic ARC-Easy
    # -------------------------------------------------------------
    echo "Evaluating base model $model_name on Indic ARC-Easy ..."

    # Indic ARC-Easy zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8

    # Indic ARC-Easy 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8
    
    # -------------------------------------------------------------
    #                       Indic ARC-Challenge
    # -------------------------------------------------------------

    echo "Evaluating base model $model_name on Indic ARC-Challenge ..."

    # Indic ARC-Challenge zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --eval_batch_size 8

    # Indic ARC-Challenge 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge-hi/$model_name-5shot" \
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
    #                       ARC-Easy
    # -------------------------------------------------------------

    echo "Evaluating model $model_name on ARC-Easy"

    # ARC-Easy zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai2_arc" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8

    # ARC-Easy 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai2_arc" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8
    
    # -------------------------------------------------------------
    #                       ARC-Challenge
    # -------------------------------------------------------------

    echo "Evaluating model $model_name on ARC-Challenge"

    # ARC-Challenge zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai2_arc" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8

    # ARC-Challenge 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai2_arc" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8
    
    # -------------------------------------------------------------
    #                       Indic ARC-Easy
    # -------------------------------------------------------------
    echo "Evaluating base model $model_name on Indic ARC-Easy ..."

    # Indic ARC-Easy zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8

    # Indic ARC-Easy 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8
    
    # -------------------------------------------------------------
    #                       Indic ARC-Challenge
    # -------------------------------------------------------------

    echo "Evaluating base model $model_name on Indic ARC-Challenge ..."

    # Indic ARC-Challenge zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge-hi/$model_name-0shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8

    # Indic ARC-Challenge 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge-hi/$model_name-5shot" \
        --model_name_or_path $model_path_or_name \
        --tokenizer_name_or_path $model_path_or_name \
        --awq \
        --eval_batch_size 8
done


# ============================================================
#                   Adapted Models
# ============================================================

for adap in $ADAPTED; do
    model_name=$(basename "${adap}")
    # -------------------------------------------------------------
    #                       ARC-Easy
    # -------------------------------------------------------------

    echo "Evaluating adapted model $model_name on ARC-Easy"
    # ARC-Easy zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai2_arc" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy/$model_name-0shot" \
        --model_name_or_path $adap \
        --tokenizer_name_or_path $adap \
        --eval_batch_size 8 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # ARC-Easy 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai2_arc" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy/$model_name-5shot" \
        --model_name_or_path $adap \
        --tokenizer_name_or_path $adap \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # -------------------------------------------------------------
    #                       ARC-Challenge
    # -------------------------------------------------------------

    echo "Evaluating adapted model $adap on ARC-Challenge"

    # ARC-Challenge zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai2_arc" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge/$model_name-0shot" \
        --model_name_or_path $adap \
        --tokenizer_name_or_path $adap \
        --eval_batch_size 8 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # ARC-Challenge 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai2_arc" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge/$model_name-5shot" \
        --model_name_or_path $adap \
        --tokenizer_name_or_path $adap \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION
    
    # -------------------------------------------------------------
    #                       Indic ARC-Easy
    # -------------------------------------------------------------
    echo "Evaluating adapted model $model_name on Indic ARC-Easy"

    # Indic ARC-Easy zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy-hi/$model_name-0shot" \
        --model_name_or_path $adap \
        --tokenizer_name_or_path $adap \
        --eval_batch_size 8 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # Indic ARC-Easy 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "easy" \
        --save_dir "$RESULTS/arc-easy-hi/$model_name-5shot" \
        --model_name_or_path $adap \
        --tokenizer_name_or_path $adap \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # -------------------------------------------------------------
    #                       Indic ARC-Challenge
    # -------------------------------------------------------------

    echo "Evaluating adapted model $model_name on Indic ARC-Challenge"

    # Indic ARC-Challenge zero-shot
    python3 -m eval.arc.run_eval \
        --ntrain 0 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge-hi/$model_name-0shot" \
        --model_name_or_path $adap \
        --tokenizer_name_or_path $adap \
        --eval_batch_size 8 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION

    # Indic ARC-Challenge 5-shot
    python3 -m eval.arc.run_eval \
        --ntrain 5 \
        --dataset "ai4bharat/ai2_arc-hi" \
        --subset "challenge" \
        --save_dir "$RESULTS/arc-challenge-hi/$model_name-5shot" \
        --model_name_or_path $adap \
        --tokenizer_name_or_path $adap \
        --eval_batch_size 4 \
        --use_chat_format \
        --chat_formatting_function $CHAT_FORMATTING_FUNCTION
done