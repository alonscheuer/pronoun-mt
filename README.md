# Exploration and Token Attribution of Context-Aware Machine Translation in German and Chinese

## Dependencies

Running the code will likely require two separate environments: one for tagging using MuDA, and one for performing LLM-reliant tasks such as fine-tuning, translating, and attribution. MuDA has very specific requirements, and so we recommend employing a separate environment for it.

Installing dependencies for the MuDA environment can be done using:
```
pip install -r muda/requirements.txt
```
MuDA dependencies may be sensitive to python versions. We recommend using python 3.9.6.

Installing dependencies for the main environment can be done using:
```
pip install -r requirements.txt
```

## Preprocessing

To prepare the data, run:
```
python collect_data.py 	--config [iwslt2017-en-de|iwslt2017-en-zh]
```
Choosing between either the `iwslt2017-en-de` config for German, or `iwslt2017-en-zh` for Chinese. The code will create the necessary files for running the MuDA tagging. This amounts to 9 files overall: for each dataset split (train, validation, test) there is one file containing English sentences (\*_muda.en), one file containing the sentences in the target language (\*_muda.de/zh), and one file containing document ids (\*_docids). These are needed as input for MuDA.

## MuDA Tagging

The [MuDA](https://github.com/CoderPat/MuDA) tagger is applied to identify examples containing context-sensitive tokens. It can be run as follows:
```
# German
python muda/main.py --src de_train_muda.en --tgt de_train_muda.de --docids de_train_muda.docids --tgt-lang de --dump-tags de_train_muda.tags --phenomena formality pronouns 

# Chinese
python muda/main.py --src zh_train_muda.en --tgt zh_train_muda.zh --docids zh_train_muda.docids --tgt-lang zh --dump-tags zh_train_muda.tags --phenomena formality pronouns classifiers
```

Make sure to install the MuDA dependencies first, preferably in a separate virtual environment.

Running the MuDA tagger will output a \*.tags file. After tagging is done, run the following code to reformat the data for the next phases:
```
python merge_tags.py --src de_train_muda.en --tgt de_train_muda.de --docids de_train_muda.docids --tags de_train_muda.tags --output de_train.json
```

## Fine-tuning the Context Aware Translation Model

The translation model can be fine-tuned as follows:
```
python finetune.py --train zh_train.json --validation zh_validation.json --base_model facebook/mbart-large-50-one-to-many-mmt --target_lang zh --max_input_length 128 --max_output_length 128 --batch_size 32 --output_dir example/my_context_aware_en_to_zh_model
```

Make sure both `train.json` and `validation.json` files have been formatted appropriately as explained above.

## Translating

To translate, run:
```
python translate.py --src zh_test.json --model example/my_context_aware_en_to_zh_model --target_land zh --add_context --device cuda --shortname zh_context_aware --output translations.json
```

Make sure the `test.json` file has been formatted appropriately using `merge_tags.py`. `--shortname` is the name displayed during the evaluation later on.

## Evaluating the Translations

To evaluate the translations, run:
```
python evaluate.py --input translations.json --output scores.csv
```

## Performing Token Attribution

To perform context attribution using [Inseq](https://github.com/inseq-team/inseq), manually extract translation examples from `translations.json`. Each example should be an object containing at least `src` and `translation` attributes. If needed, optional attributes of `contrast` and `aligment` can be added for contrast attribution. The input file should be a JSON file containing an array of the examples you wish to insepct. Then run:
```
python attribute.py --input my_examples.json --model example/my_context_aware_en_to_zh_model --target_lang zh
```