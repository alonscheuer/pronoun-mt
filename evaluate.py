from sacrebleu.metrics import BLEU, CHRF
import argparse
import json
import pandas as pd
from comet import download_model, load_from_checkpoint


bleu = BLEU()
chrf = CHRF()
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    inputs = json.load(open(args.input))

    grouped = {}
    for sample in inputs:
        for tag in sample["tags"]:
            if not tag in grouped:
                grouped[tag] = []
            grouped[tag].append(sample["translations"])

    inputs = grouped

    phenomena = inputs.keys()
    results = pd.DataFrame({"translation": ["{}-{}".format(p, translation_model) for p in phenomena for translation_model in inputs[p]]})

    Bleu = []
    ChrF = []
    Comet = []

    for p in phenomena:
        for tm in p:
            src = [sent["src"] for sent in tm]
            ref = [sent["tgt"] for sent in tm]
            translation = [sent["translation"] for sent in tm]

            Bleu.append(bleu.corpus_score(translation, [ref]))
            ChrF.append(chrf.corpus_score(translation, [ref]))
            Comet.append(comet_model.predict([{"src": s, "mt": m, "ref": r}
                                              for s, m, r in zip(src, translation, ref)],
                                       batch_size=8,
                                       gpus=1).system_score)

            results["Bleu"] = Bleu
            results["ChrF"] = ChrF
            results["Comet"] = Comet
            results.to_csv(args.output)


if __name__ == "__main__":
    main()
