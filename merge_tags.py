import argparse
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=str)
    parser.add_argument("--tgt", required=True, type=str)
    parser.add_argument("--docids", required=True, type=str)
    parser.add_argument("--tags", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def group_by_doc(sents, docids):
    result = []
    old_doc = 0
    doc_sents = []
    for i_sent, sent in enumerate(sents):
        new_doc = docids[i_sent]
        # if reached a new doc
        if new_doc != old_doc:
            if len(doc_sents) > 0:
                # append old doc sentences to result
                result.append(doc_sents)
            # initialize a new doc
            doc_sents = []
            old_doc = new_doc
        doc_sents.append(sent)
    return result


def main():
    args = parse_args()

    samples = []

    tag_docs = json.load(open(args.tags))

    with open(args.docids) as file:
        docids = file.read().splitlines()

    with open(args.src) as file:
        srcs = file.read().splitlines()
    src_docs = group_by_doc(srcs, docids)

    with open(args.tgt) as file:
        tgts = file.read().splitlines()
    tgt_docs = group_by_doc(tgts, docids)
    
    for tag_doc, src_doc, tgt_doc in tqdm(zip(tag_docs, src_docs, tgt_docs)):
        for i_sent, sent in enumerate(tag_doc):
            if any([len(tok["tags"]) > 0 for tok in sent]) and i_sent > 0:
                tags = list(set([tag for tok in sent for tag in tok["tags"]]))
                samples.append({"src": src_doc[i_sent],
                                "ctx_src": src_doc[i_sent - 1],
                                "tgt": tgt_doc[i_sent],
                                "ctx_tgt": tgt_doc[i_sent - 1],
                                "tags": tags})

    json.dump(samples, open(args.output), indent=2)


if __name__ == "__main__":
    main()
