# export CUDA_VISIBLE_DEVICES=0


model_name_or_path="sarvamai/OpenHathi-7B-Hi-v0.1-Base"

echo "evaluating openhathi base on hatecheck-hi ..."

# zero-shot
python3 -m eval.hatecheckhin.run_eval \
    --ntrain 0 \
    --save_dir "results/hatecheckhin/openhathi-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1

# 5-shot
python3 -m eval.hatecheckhin.run_eval \
    --ntrain 5 \
    --save_dir "results/hatecheckhin/openhathi-base-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on hatecheck-hi ..."

# zero-shot
python3 -m eval.hatecheckhin.run_eval \
    --ntrain 0 \
    --save_dir "results/hatecheckhin/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function $CHAT_FORMATTING_FUNCTION


# 5-shot
python3 -m eval.hatecheckhin.run_eval \
    --ntrain 5 \
    --save_dir "results/hatecheckhin/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function $CHAT_FORMATTING_FUNCTION
