from typing import Any
import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("de_tagger")
class GermanTagger(Tagger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.formality_classes = {
            "t_class": {
                "deiner",
                "dich",
                "dir",
                "du",
                "er",
                "es",
                "euch",
                "eurer",
                "ihm",
                "ihn",
                "ihnen",
                "ihr",
                "ihrer",
                "seiner",
                "sie"
            },
            "v_class": {
                "Ihm",
                "Ihn",
                "Ihnen",
                "Ihr",
                "Ihrer",
                "Seiner",
                "Sie"
            }
        }
        from spacy.lang.de.stop_words import STOP_WORDS

        self.ambiguous_pronouns = {
                "he": ["er", "Sie"],
                "her": ["sie", "Sie", "ihr", "Ihr", "ihrer", "Ihrer"],
                "him": ["ihn", "Ihn", "ihm", "Ihm"],
                "his": ["seiner", "Seiner"],
                "it": ["es", "ihm"],
                "she": [ "sie", "Sie"],
                "their": ["ihrer", "Ihrer"],
                "them": ["sie", "ihnen", "Sie", "Ihnen"],
                "they": ["sie", "Sie"],
                "you": ["euch", "Sie", "Ihnen", "ihr", "dich", "dir", "du"],
                "your": ["eurer", "Ihrer", "Sie", "deiner"]
        }

        self.stop_words = STOP_WORDS
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "de", processors="tokenize,pos,lemma,depparse"
        )
