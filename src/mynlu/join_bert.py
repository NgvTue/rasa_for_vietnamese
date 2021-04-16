import os
import typing
from typing import Any, Optional, Text, Dict, List, Type


import numpy as np
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.featurizers.featurizer import DenseFeaturizer

from rasa.nlu.featurizers.dense_featurizer import convert_featurizer
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)
import logging
if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
def load_model(path=None):
    if path is None:
        path = 'vinai/phobert-base'
    phobert = AutoModel.from_pretrained(path)
    # For transformers v4.x+: 
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    with open("ok2.txt","w+") as f:
        f.write("okkkk")
    return phobert, tokenizer

class FastTextFeaturizer(DenseFeaturizer):
    """This component adds fasttext features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["torch"]

    defaults = {"path": None, "device": 'cpu'}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        
        path = component_config["path"]

        self.device = component_config.get("device",'cpu')
        self.model,self.tokenizer = load_model(path)
        logging.debug("load vinai done")
        self.model = self.model.to(self.device)
        
    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        for example in training_data.training_examples:
            for attribute in (DENSE_FEATURIZABLE_ATTRIBUTES):
                self.gen_feature(example, attribute)

   
    def gen_feature(self, message, attribute = TEXT):
        tokens = message.get(TOKENS_NAMES[attribute])
        
        if not tokens:
            logging.debug("tokens is none at process")
            return None
        
        tokens = [t.text for t in tokens]
        cache_word = []
        len_word = []
        unk_token = self.tokenizer.unk_token
        sep_token = self.tokenizer.sep_token
        cls_token = self.tokenizer.cls_token
        for token in tokens:
            # with open("./ok1.txt","w+") as f:
            #     f.write(str(token))
            word_tokens = self.tokenizer.tokenize(token)
            if not word_tokens:
                word_tokens = [unk_token]
            cache_word = cache_word + word_tokens
            len_word.append(
                len(word_tokens)
            )
        cache_word += [sep_token]
        cache_word = [cls_token] + cache_word
        # encoder done
        tensor_inputs = self.tokenizer.convert_tokens_to_ids(cache_word)
        tensor_inputs = torch.tensor(
            tensor_inputs, dtype=torch.long
        ).unsqueeze(0)
        tensor_inputs = tensor_inputs.to(self.device)
        with torch.no_grad():
            out_vectors = self.model(tensor_inputs)
        
        word_feature = out_vectors[0].squeeze(0).to("cpu").numpy()
        
        sentence_feature = out_vectors[1].to("cpu").numpy().reshape(1,-1)
        st = 0
        word_reduce = []
        for i in range(len(len_word)):
            l = len_word[i]
            en = st + l
            a1 = word_feature[st+1:en+1,...]
            a1 = np.mean(a1, axis = 0)
            word_reduce.append(
                a1
            )
            st = en

        assert len(word_reduce) == len(tokens) 
       
        word_reduce = np.array(word_reduce).reshape(len(tokens), -1)
        final_sequence_features = Features(
            word_reduce,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            sentence_feature,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    def process(self, message: Message, **kwargs: Any) -> None:
        
        self.gen_feature(
            message,
            attribute=TEXT
        )
        
        
    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        return None

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component

        return cls(meta)