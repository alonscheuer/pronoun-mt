import inseq
from inseq.data.aggregator import SubwordAggregator
import argparse
import json


lang_codes = {
    "de": "de_DE",
    "zh": "zh_CN"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--target_lang", required=True, type=str)
    return parser.parse_args()


def regular_attribution(model, prompt, target, lang_code, aggregate=False):
    out = model.attribute(
        prompt,
        target,
        attribute_target=True,
        step_scores=["probability"],
        generation_args={"forced_bos_token_id": model.tokenizer.lang_code_to_id[lang_code]},
    )

    if aggregate:
        out.aggregate().show(aggregator=SubwordAggregator)
    else:
        out.show()


def contrast_attribution(model, prompt, target, contrast_target, lang_code, alignment=None, aggregate=False):
    out = model.attribute(
        prompt,
        # Fix the original target
        target,
        attribute_target=True,
        attributed_fn="contrast_prob_diff",
        # Also show the probability delta between the two options
        step_scores=["contrast_prob_diff", "probability"],
        contrast_targets=contrast_target,
        contrast_targets_alignments=alignment,
        generation_args={"forced_bos_token_id": model.tokenizer.lang_code_to_id[lang_code]},
    )

    if aggregate:
        out.aggregate().show(aggregator=SubwordAggregator)
    else:
        out.show()


def main():
    args = parse_args()

    lang_code = lang_codes.get(args.target_lang)

    model = inseq.load_model(
        args.model,
        "input_x_gradient",
        # The tokenizer_kwargs are used to specify the source and target languages upon initialization
        tokenizer_kwargs={"src_lang": "en_XX", "tgt_lang": lang_code},
    )

    inputs = json.load(open(args.input))

    for sample in inputs:
        regular_attribution(model,
                            sample["src"],
                            sample["translation"],
                            lang_code,
                            args.aggregate)

        if sample["contrast"]:
            contrast_attribution(model,
                                 sample["src"],
                                 sample["translation"],
                                 sample["contrast"],
                                 lang_code,
                                 sample.get("alignment"),
                                 args.aggregate)


if __name__ == "__main__":
    main()


