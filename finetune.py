import argparse
from transformers import AutoTokenizer, MBartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, load_metric
import json
import numpy as np


lang_codes = {
		"de": "de_DE",
		"zh": "zh_CN"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=str, help="File containing train set")
    parser.add_argument("--validation", required=True, type=str, help="File containing validation set")
    parser.add_argument("--base_model", required=True, type=str, help="Base translation model to use")
    parser.add_argument("--target_lang", required=True, type=str, help="Target language code")
    parser.add_argument("--max_input_length", required=True, type=int, help="Maximum input length")
    parser.add_argument("--max_output_length", required=True, type=int, help="Maximum output length")
    parser.add_argument("--batch_size", required=True, type=int, help="Fine-tuning batch size")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to save training data and model")
    return parser.parse_args()


def create_dataset(d):
    ctx_src = [e["ctx_src"] for e in d]
    src = [e["src"] for e in d]
    ctx_tgt = [e["ctx_tgt"] for e in d]
    tgt = [e["tgt"] for e in d]

    ds = Dataset.from_dict({
        "ctx_src": ctx_src,
        "src": src,
        "ctx_tgt": ctx_tgt,
        "tgt": tgt
    })
    return ds


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds, tokenizer):
    metric = load_metric("sacrebleu")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def preprocess(es, tokenizer, max_input_length, max_output_length):
    input = [ctx_src + " <brk> " + src for ctx_src, src in zip(es["ctx_src"], es["src"])]
    target = [ctx_tgt + " <brk> " + tgt for ctx_tgt, tgt in zip(es["ctx_tgt"], es["tgt"])]
    model_inputs = tokenizer(input, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, max_length=max_output_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    args = parse_args()
    dat_train = json.load(open(args.train))
    dat_eval = json.load(open(args.validation))

    train_raw = create_dataset(dat_train)
    eval_raw = create_dataset(dat_eval)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.src_lang = "en-XX"
    tokenizer.tgt_lang = lang_codes.get(args.target_lang)

    train_set = train_raw.map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer,
                                                                   "max_input_length": args.max_input_length,
                                                                   "max_output_length": args.max_output_length})
    eval_set = eval_raw.map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer,
                                                                   "max_input_length": args.max_input_length,
                                                                   "max_output_length": args.max_output_length})

    model = MBartForConditionalGeneration.from_pretrained(args.base_model)

    training_args = Seq2SeqTrainingArguments(
        args.output_dir,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == "__main__":
    main()
