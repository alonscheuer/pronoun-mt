from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import json
from tqdm import tqdm
import argparse


lang_codes = {
    "de": "de_DE",
    "zh": "zh_CN"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--target_lang", required=True, type=str)
    parser.add_argument("--add_context", action="store_true")
    parser.add_argument("--device", required=True, type=str)
    parser.add_argument("--shortname", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = json.load(open(args.src))
    model = MBartForConditionalGeneration.from_pretrained(args.model).to(args.device)
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model, src_lang="en_XX")

    translations = []
    for sample in tqdm(samples):
        src = sample["ctx_src"] + " <brk> " + sample["src"] if args.add_context else sample["src"]
        inputs = tokenizer(src, return_tensors = "pt").to(args.device)
        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_codes.get(args.target_lang)])
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sample["translations"][args.shortname]["src"] = src
        sample["translations"][args.shortname]["tgt"] = sample["tgt"]
        sample["translations"][args.shortname]["translation"] = translation

    with open(args.output, "w") as output:
        json.dump(samples, output)


if __name__ == "__main__":
    main()
