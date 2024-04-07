from typing import Any
import spacy_stanza  # type: ignore

from tagger import Tagger

from . import register_tagger


@register_tagger("zh_tagger")
class ChineseTagger(Tagger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        self.formality_classes = {
            "t_class": {"你"},
            "v_class": {"您"},
        }
        
        self.ambiguous_pronouns = {
            "they": {"他们", "她们", "它们"},
            "them": {"他们", "她们", "它们"},
        }

        self.quantifier_set = {"个", "只", "张", "位", "条", "本", "块", "把", "头", "朵", "匹", "棵", "颗", "件", "台", "辆", "艘", "架", "层", "排", "组", "批", "堆", "束", "盒", "袋", "瓶", "罐", "桶", "盘", "碗", "碟", "双", "对", "顶", "支", "根", "丝", "名"}
        # self.tgt_pipeline = spacy.load("zh_core_web_sm")
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "zh", processors="tokenize,pos,lemma,depparse"
        )
