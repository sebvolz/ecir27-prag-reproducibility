import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import utils
from root_dir_path import ROOT_DIR
from utils import load_data

def main(args):
    utils.DATA_ROOT_DIR = os.path.join(ROOT_DIR, args.data_dir)
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    tokenizer = AutoTokenizer.from_pretrained(args.cross_encoder)
    model = AutoModelForSequenceClassification.from_pretrained(args.cross_encoder).cuda().eval()
    output_dir = os.path.join(ROOT_DIR, args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        ret = []
        for data in tqdm(fulldata):
            inputs = tokenizer([data["question"]] * len(data["passages"]), data["passages"], padding=True,
                               truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.no_grad():
                ret.append(model(**inputs).logits[:, 0].tolist())
        with open(os.path.join(output_dir, f"{filename}.json"), "w") as fout:
            json.dump(ret, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--augment_model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data_aug_top20")
    parser.add_argument("--output_dir", type=str, default="data_relevance")
    parser.add_argument("--cross_encoder", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()
    print(args)
    main(args)
