import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from eval.utils import (
    get_next_word_predictions,
    load_hf_lm_and_tokenizer,
    dynamic_import_function,
)

choices = ["positive", "negative"]
choices_to_id = {choice: i for i, choice in enumerate(choices)}


def format_example(text, label=None):
    prompt = "Review: {text}".format(text=text)
    prompt += "\nSentiment:"
    if label is not None:
        prompt += " {label}\n\n".format(label=label)
    return prompt


def gen_prompt(dev_data, k=-1):
    prompt = "Predict the sentiment of the review. The possible choices for the sentiment are: 'positive' and 'negative'.\n\n"
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            prompt += format_example(text=example["INDIC REVIEW"], label=example["LABEL"])
    return prompt


def main(args):
    random.seed(args.seed)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq, 
            awq_model=args.awq, 
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset("ai4bharat/IndicSentiment", f"translation-{args.lang}")
    dataset = dataset.filter(lambda x: x["LABEL"] is not None)
    dataset = dataset.map(lambda x: {"LABEL": x["LABEL"].lower().strip()})
    dev_data = dataset["validation"].shuffle(args.seed)
    test_data = dataset["test"]

    prompts = []
    for i, example in enumerate(test_data):
        k = args.ntrain
        prompt_end = format_example(example["INDIC REVIEW"])
        train_prompt = gen_prompt(dev_data, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is: "
            else:
                prompt += " The answer is: "

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_data, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is: "
                else:
                    prompt += " The answer is: "

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    # get the answer for all examples
    # adding a prefix space here, as that's expected from the prompt
    # TODO: should raise a warning if this returns more than one token
    answer_choice_ids = [tokenizer.encode(answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model,
        tokenizer,
        prompts,
        candidate_token_ids=answer_choice_ids,
        return_token_predictions=False,
        batch_size=args.eval_batch_size,
    )

    # get the metrics
    ground_truths = [choices_to_id[example["LABEL"]] for example in test_data]
    predictions = [pred_index for pred_index in pred_indices]
    metrics = {
        "accuracy": accuracy_score(ground_truths, predictions),
        "precision": precision_score(ground_truths, predictions),
        "recall": recall_score(ground_truths, predictions),
        "f1": f1_score(ground_truths, predictions),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=[
            "as",
            "bn",
            "gu",
            "hi",
            "kn",
            "ml",
            "mr",
            "or",
            "pa",
            "ta",
            "te",
            "ur",
        ],
    )
    parser.add_argument("--save_dir", type=str, default="results/indicsentiment/llama-7B/")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--awq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized AWQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    args = parser.parse_args()
    main(args)
