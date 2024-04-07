import argparse
from datasets import load_dataset
from tqdm import tqdm


configs = {
    "de": "iwslt2017-en-de",
    "zh": "iwslt2017-en-zh"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset("gsarti/iwslt2017_context", configs[args.config])
    for split in tqdm(dataset):
        split_ds = dataset[split].to_pandas()
        srcs = [t.get("en") for t in split_ds["translation"]]
        tgts = [t.get(args.config) for t in split_ds["translation"]]
        with open("{}_{}_muda.en".format(args.config, split), "w") as file:
            file.writelines(src + "\n" for src in srcs)
        with open("{}_{}_muda.{}".format(args.config, split, args.config), "w") as file:
            file.writelines(tgt + "\n" for tgt in tgts)
        with open("{}_{}_muda.docids".format(args.config, split), "w") as file:
            file.writelines(str(doc_id) + "\n" for doc_id in split_ds["doc_id"])


if __name__ == "__main__":
    main()
