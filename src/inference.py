import os
import gc
import json
import argparse
import torch
import numpy as np
import peft
from tqdm import tqdm
from peft import PeftModel

import prompt_template
import utils
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete

def get_merge_weights(args, n, scores=None):
    if args.merge_weight == "relevance":
        z = np.array(scores[:n])
        w = np.exp(z - z.max())
        return (w / w.sum()).tolist()
    return [{"one": 1, "inv_n": 1 / n, "inv_n2": 1 / n ** 2}[args.merge_weight]] * n


def main(args):
    utils.DATA_ROOT_DIR = os.path.join(ROOT_DIR, args.data_dir)
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    if args.num_passages:
        for _, fulldata in data_list:
            for data in fulldata:
                assert len(data["passages"]) >= args.num_passages
                data["passages"] = data["passages"][:args.num_passages]
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    cot_name = "cot" if args.with_cot else "direct"
    load_adapter_path = os.path.join(
        ROOT_DIR,
        args.offline_dir,
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output",
        *([args.exp_name] if args.exp_name else []),
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method, 
    )
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump({**vars(args), "peft_version": peft.__version__}, fout, indent=4)

        if args.merge_weight == "relevance":
            with open(os.path.join(ROOT_DIR, args.relevance_dir, args.dataset, f"{filename}.json")) as fin:
                relevance_scores = json.load(fin)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        if args.shuffle_adapters or args.shuffle_passages:
            # question i uses the adapters / passages of question donor[i] != i
            perm = np.random.RandomState(args.shuffle_seed).permutation(len(fulldata))
            donor = np.empty(len(fulldata), dtype=int)
            donor[perm] = np.roll(perm, -1)
        alldata = fulldata
        fulldata = fulldata[start_with:]
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs):
                text = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, 
                                        passages=psgs)
                pred = {
                    "test_id": test_id, 
                    "question": question, 
                    "answer": answer, 
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            if args.inference_method == "icl":
                passage_id = int(donor[test_id]) if args.shuffle_passages else test_id
                ret.append(get_pred(model, psgs=alldata[passage_id]["passages"]))
                ret[-1]["passage_id"] = passage_id
            elif args.inference_method == "vanilla":
                ret.append(get_pred(model, psgs=None))
            else:
                adapter_id = int(donor[test_id]) if args.shuffle_adapters else test_id
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{adapter_id}", f"passage_{pid}")
                    if pid == 0:
                        model = PeftModel.from_pretrained(
                            model, 
                            adapter_path,
                            adapter_name = "0", 
                            is_trainable = False
                        )
                    else:
                        model.load_adapter(adapter_path, adapter_name = str(pid)) 
                # merge
                if args.merge_strategy == "dare_linear":
                    torch.manual_seed(args.merge_seed + test_id)
                scores = relevance_scores[test_id] if args.merge_weight == "relevance" else None
                model.add_weighted_adapter(
                    adapters = [str(i) for i in range(len(passages))],
                    weights = get_merge_weights(args, len(passages), scores),
                    adapter_name = "merge",
                    combination_type = args.merge_strategy,
                    density = args.merge_density if args.merge_strategy in ["ties", "dare_linear"] else None,
                )
                model.set_adapter("merge")
                ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
                ret[-1]["adapter_id"] = adapter_id
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)  
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["vanilla", "icl", "prag", "combine"])
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data_aug")
    parser.add_argument("--offline_dir", type=str, default="offline")
    parser.add_argument("--num_passages", type=int, default=None) # use only the top-n passages
    parser.add_argument("--shuffle_adapters", action="store_true")
    parser.add_argument("--shuffle_passages", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    # Merging
    parser.add_argument("--merge_strategy", type=str, default="cat", choices=["cat", "linear", "ties", "dare_linear"])
    parser.add_argument("--merge_weight", type=str, default="one", choices=["one", "inv_n", "inv_n2", "relevance"])
    parser.add_argument("--merge_density", type=float, default=0.5) # ties, dare_linear
    parser.add_argument("--merge_seed", type=int, default=0) # dare_linear
    parser.add_argument("--relevance_dir", type=str, default="data_relevance") # scores from score_passages.py
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    assert not args.shuffle_adapters or args.inference_method in ["prag", "combine"]
    assert not args.shuffle_passages or args.inference_method == "icl"
    assert args.merge_weight != "relevance" or (args.merge_strategy == "cat" and not args.shuffle_adapters)
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)