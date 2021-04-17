import os
import typing
from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type
import scipy.sparse
import tqdm
import logging
import numpy as np
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
import rasa.nlu.utils.bilou_utils as bilou_utils
import copy
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import os
import scipy.sparse
import tensorflow as tf

from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type

import rasa.shared.utils.io
import rasa.utils.io as io_utils
import rasa.nlu.utils.bilou_utils as bilou_utils
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.components import Component
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.extractors.extractor import EntityExtractor, EntityTagSpec
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.nlu.featurizers.dense_featurizer import convert_featurizer
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
    
)
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureSignature,
    FeatureArray,
)
from rasa.utils.tensorflow.constants import (
    LABEL,
    IDS,
    HIDDEN_LAYERS_SIZES,
    SHARE_HIDDEN_LAYERS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    TENSORBOARD_LOG_DIR,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    WEIGHT_SPARSITY,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    AUTO,
    BALANCED,
    CROSS_ENTROPY,
    TENSORBOARD_LOG_LEVEL,
    CONCAT_DIMENSION,
    FEATURIZERS,
    CHECKPOINT_MODEL,
    SEQUENCE,
    SENTENCE,
    SEQUENCE_LENGTH,
    DENSE_DIMENSION,
    MASK,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    SOFTMAX,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    EXTRACTOR,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    NO_ENTITY_TAG,
    SPLIT_ENTITIES_BY_COMMA,
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
    SINGLE_ENTITY_ALLOWED_INTERLEAVING_CHARSET,
)
import logging
if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata
logger = logging.getLogger(__name__)


def collated_fn(batch):
    return [i for i in zip(*batch)]

SPARSE = "sparse"
DENSE = "dense"
LABEL_KEY = LABEL
LABEL_SUB_KEY = IDS

POSSIBLE_TAGS = [ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP]
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
def load_model(path=None, cache_dir=None):
    if path is None:
        path = 'vinai/phobert-base'
    phobert = AutoModel.from_pretrained(path, cache_dir=cache_dir, local_files_only=True)
    # For transformers v4.x+: 
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
    return phobert, tokenizer

class Bert(DenseFeaturizer):
    """This component adds fasttext features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["torch"]

    defaults = {"path": None, "device": 'cpu',"cache_dir":'models',BILOU_FLAG: True,FEATURIZERS: [],
                INTENT_CLASSIFICATION: True,
                    # If 'True' named entity recognition is trained and entities predicted.
                ENTITY_RECOGNITION: True,}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        
        path = component_config["path"]
        cache_dir = component_config.get("cache_dir","models")
        self.device = component_config.get("device",'cpu')
        self.model,self.tokenizer = load_model(path, cache_dir)
        logging.debug("load vinai done")
        self.model = self.model.to(self.device)
    def filter_trainable_entities(
        self, entity_examples: List[Message]
    ) -> List[Message]:
        """Filters out untrainable entity annotations.

        Creates a copy of entity_examples in which entities that have
        `extractor` set to something other than
        self.name (e.g. 'CRFEntityExtractor') are removed.
        """

        filtered = []
        z= 0 
        for message in entity_examples:
            entities = []
            for ent in message.get(ENTITIES, []):
                extractor = ent.get(EXTRACTOR)
                if not extractor or extractor == self.name:
                    entities.append(ent)
            data = message.data.copy()
            data[ENTITIES] = entities
            
            filtered.append(
                Message(
                    text=message.get(TEXT),
                    data=data,
                    output_properties=message.output_properties,
                    time=message.time,
                    features=message.features,
                )
            )

        return filtered  
    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        # log self.index_label_id_mapping
        data = self.process_data_customize(training_data, config, **kwargs)
        entity_examples = self.filter_trainable_entities(training_data.nlu_examples)
        # todos add more ENTITY_ATTR
        entities_train = data.get(
            ENTITIES,
            ENTITY_ATTRIBUTE_TYPE
        )[0] 
        # entities_train = [i]
        logging.debug(
            f"{str(entities_train[0,...])} - {len(entities_train)} - {isinstance(entities_train, np.ndarray)}\n- {entities_train[0].shape}"
        )
        entities_train = [i.astype(np.int32).reshape(-1).tolist() for i in entities_train] 
        intents_train = data.get(
            LABEL,
            IDS
        )[0]
        intents_train = [torch.tensor(i.astype(np.int32), dtype=torch.int32) for i in intents_train]
        # logging.debug(f"{str(intents_train[0])} - {len(intents_train)} - {len(intents_train[0])}")
        logging.debug(f"{self._entity_tag_specs[0].tags_to_ids['O']}")
        inputs = []
        cache_len = []
        mask = []
        index = 0
        pad_ent = self._entity_tag_specs[0].tags_to_ids['O']
        for example in tqdm.tqdm(training_data.training_examples):
            tokens = example.get(TOKENS_NAMES[TEXT])
            if tokens:
                inc, cache = self.process_inputs_signature(
                    tokens,
                    device='cpu'
                )
                inputs.append(
                    inc
                )
                cache_len.append(
                    cache
                )
                entities_example = [pad_ent] 
                for l,x in zip(cache,entities_train[index]):
                    entities_example = entities_example + [x,] * l
                entities_example  = entities_example + [pad_ent]
                # if index == 0:
                #     logging.debug(f'{cache} - {entities_train[index]} - {entities_example} - {[t.text for t in tokens]}')
                entities_train[index] = entities_example
                index += 1


        pad_s = max([int(i.size(0))  for i in inputs])
        for i in range(len(inputs)):
            s = inputs[i].size(0)
            ma = [1,] * s
            if s < pad_s:
                inputs[i] = torch.cat(
                    (inputs[i], torch.zeros(pad_s - s)
                ))
                ma = ma + [0,] * (pad_s -s)
            s_e = len(entities_train[i])
            if s_e < pad_s:
                entities_train[i] = entities_train[i] + [pad_ent,] * (pad_s-s_e)
            entities_train[i] = torch.tensor(entities_train[i], dtype=torch.int32)  
            mask.append(torch.tensor(ma,dtype=torch.long))
        inputs = torch.stack(
            inputs
        )
        entities_train = torch.stack(
            entities_train
        )
        intents_train = torch.stack(
            intents_train
        )
        mask = torch.stack(
            mask
        )
        logging.debug(f"{inputs.size()}- {pad_s * len(cache_len)} - {intents_train.size()} - {entities_train.size()}")
        logging.debug(f"{intents_train.size()} - {mask.size()} - {entities_train.size()}")
        assert int(inputs.size(0) * inputs.size(1)) == pad_s * len(cache_len)
        dataset_tensor = torch.utils.data.TensorDataset(
            inputs,
            mask,
            entities_train,
            intents_train
        )
        data_loader = torch.utils.data.DataLoader(
            dataset_tensor,
            batch_size=32,
            shuffle=True,
            collate_fn=collated_fn,
            num_workers=2
        )
        logging.debug(f"prepare data done")
        

        # todos : join training and save_respone-actions to speed up training
        for example in tqdm.tqdm(training_data.training_examples):
            for attribute in (DENSE_FEATURIZABLE_ATTRIBUTES):
                self.gen_feature(example, attribute)

    def process_inputs_signature(
        self, 
        tokens,
        device
    ):
        
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
            # logging.debug(f"{token}  - {word_tokens} - xx {len(len_word)}")
    
        cache_word += [sep_token]
        cache_word = [cls_token] + cache_word
        # logging.debug(f"{cache_word} - {len_word} - {len(len_word)}")
        # encoder done
        tensor_inputs = self.tokenizer.convert_tokens_to_ids(cache_word)
        tensor_inputs = torch.tensor(
            tensor_inputs, dtype=torch.long
        )
        tensor_inputs = tensor_inputs.to(device)
        return tensor_inputs, len_word

    def gen_feature(self, message, attribute = TEXT):
        
        tokens = message.get(TOKENS_NAMES[attribute])
        
        if not tokens:
            logging.debug("tokens is none at process")
            return None
        
        tensor_inputs, len_word = self.process_inputs_signature(
            tokens,
            self.device
        )
        
        with torch.no_grad():
            out_vectors = self.model(tensor_inputs.unsqueeze(0))

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

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[FeatureArray]:
        all_label_features = self._label_data.get(LABEL, SENTENCE)[0]
        return [
            FeatureArray(
                np.array([all_label_features[label_id] for label_id in label_ids]),
                number_of_dimensions=all_label_features.number_of_dimensions,
            )
        ]
    def _add_label_features(
        self,
        model_data: RasaModelData,
        training_data: List[Message],
        label_attribute: Text,
        label_id_dict: Dict[Text, int],
        training: bool = True,
    ) -> None:
        label_ids = []
        if training and self.component_config[INTENT_CLASSIFICATION]:
            for example in training_data:
                if example.get(label_attribute):
                    label_ids.append(label_id_dict[example.get(label_attribute)])

            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            model_data.add_features(
                LABEL_KEY,
                LABEL_SUB_KEY,
                [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
            )

        if (
            label_attribute
            and model_data.does_feature_not_exist(label_attribute, SENTENCE)
            and model_data.does_feature_not_exist(label_attribute, SEQUENCE)
        ):
            # no label features are present, get default features from _label_data
            model_data.add_features(
                LABEL, SENTENCE, self._use_default_label_features(np.array(label_ids))
            )

        # as label_attribute can have different values, e.g. INTENT or RESPONSE,
        # copy over the features to the LABEL key to make
        # it easier to access the label features inside the model itself
        model_data.update_key(label_attribute, SENTENCE, LABEL, SENTENCE)
        model_data.update_key(label_attribute, SEQUENCE, LABEL, SEQUENCE)
        model_data.update_key(label_attribute, MASK, LABEL, MASK)

        model_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)
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


    def process_data_customize(self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ):

        if self.component_config[BILOU_FLAG]:
            '''user bilou: I-U-L entities for handle group entities
            '''
            bilou_utils.apply_bilou_schema(training_data)

        label_id_index_mapping = self._label_id_index_mapping(
            training_data,
            attribute=INTENT
        )

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()
        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)
        with open("logs/intent.txt","w+") as f:
            f.write(
                str(self.index_label_id_mapping)
            )
        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=INTENT
        )
        logging.debug(f"{self._label_data.items()}")
        logging.debug("ok process")
        with open("logs/proce.txt","w+") as f:
            f.write(str(self._label_data))
        self._entity_tag_specs = self._create_entity_tag_specs(training_data)
        with open("logs/x.txt","w") as f:
            f.write(str(self._entity_tag_specs) + "\n")
            f.write("")
        label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )

        model_data = self._create_model_data(
            training_data.nlu_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
        )
        self._check_input_dimension_consistency(model_data)
        return model_data
    @property
    def label_sub_key(self) -> Optional[Text]:
        """Return sub key if intent classification is activated."""
        return LABEL_SUB_KEY if self.component_config[INTENT_CLASSIFICATION] else None
    @property
    def label_key(self) -> Optional[Text]:
        """Return key if intent classification is activated."""
        return LABEL_KEY if self.component_config[INTENT_CLASSIFICATION] else None
    def _check_input_dimension_consistency(self, model_data: RasaModelData) -> None:
        """Checks if features have same dimensionality if hidden layers are shared."""
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            num_text_sentence_features = model_data.number_of_units(TEXT, SENTENCE)
            num_label_sentence_features = model_data.number_of_units(LABEL, SENTENCE)
            num_text_sequence_features = model_data.number_of_units(TEXT, SEQUENCE)
            num_label_sequence_features = model_data.number_of_units(LABEL, SEQUENCE)

            if (0 < num_text_sentence_features != num_label_sentence_features > 0) or (
                0 < num_text_sequence_features != num_label_sequence_features > 0
            ):
                raise ValueError(
                    "If embeddings are shared text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )
    def _create_model_data(
        self,
        training_data: List[Message],
        label_id_dict: Optional[Dict[Text, int]] = None,
        label_attribute: Optional[Text] = None,
        training: bool = True,
    ) -> RasaModelData:
        """Prepare data for training and create a RasaModelData object."""
        from rasa.utils.tensorflow import model_data_utils

        attributes_to_consider = [TEXT]
        if training and self.component_config[INTENT_CLASSIFICATION]:
            # we don't have any intent labels during prediction, just add them during
            # training
            attributes_to_consider.append(label_attribute)
        if (
            training
            and self.component_config[ENTITY_RECOGNITION]
            and self._entity_tag_specs
        ):
            # Add entities as labels only during training and only if there was
            # training data added for entities with DIET configured to predict entities.
            attributes_to_consider.append(ENTITIES)

        if training and label_attribute is not None:
            # only use those training examples that have the label_attribute set
            # during training
            training_data = [
                example for example in training_data if label_attribute in example.data
            ]

        if not training_data:
            # no training data are present to train
            return RasaModelData()

        features_for_examples = model_data_utils.featurize_training_examples(
            training_data,
            attributes_to_consider,
            entity_tag_specs=self._entity_tag_specs,
            featurizers=self.component_config[FEATURIZERS],
            bilou_tagging=self.component_config[BILOU_FLAG],
        )
        attribute_data, _ = model_data_utils.convert_to_data_format(
            features_for_examples, consider_dialogue_dimension=False
        )

        model_data = RasaModelData(
            label_key=self.label_key, label_sub_key=self.label_sub_key
        )
        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)

        self._add_label_features(
            model_data, training_data, label_attribute, label_id_dict, training
        )

        # make sure all keys are in the same order during training and prediction
        # as we rely on the order of key and sub-key when constructing the actual
        # tensors from the model data
        model_data.sort()

        return model_data
    def _create_entity_tag_specs(
        self, training_data: TrainingData
    ) -> List[EntityTagSpec]:
        """Create entity tag specifications with their respective tag id mappings."""

        _tag_specs = []

        for tag_name in POSSIBLE_TAGS:
            if self.component_config[BILOU_FLAG]:
                tag_id_index_mapping = bilou_utils.build_tag_id_dict(
                    training_data, tag_name
                )
            else:
                tag_id_index_mapping = self._tag_id_index_mapping_for(
                    tag_name, training_data
                )

            if tag_id_index_mapping:
                _tag_specs.append(
                    EntityTagSpec(
                        tag_name=tag_name,
                        tags_to_ids=tag_id_index_mapping,
                        ids_to_tags=self._invert_mapping(tag_id_index_mapping),
                        num_tags=len(tag_id_index_mapping),
                    )
                )

        return _tag_specs
    @staticmethod
    def _label_id_index_mapping(
        training_data: TrainingData, attribute: Text
    ) -> Dict[Text, int]:
        """Create label_id dictionary."""

        distinct_label_ids = {
            example.get(attribute) for example in training_data.intent_examples
        } - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _invert_mapping(mapping: Dict) -> Dict:
        return {value: key for key, value in mapping.items()}

    def _create_label_data(
        self,
        training_data: TrainingData,
        label_id_dict: Dict[Text, int],
        attribute: Text,
    ) :
        """Create matrix with label_ids encoded in rows as bag of words.

        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """
        # Collect one example for each label
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]
        with open("logs/al.txt","w+") as f:
            f.write(str(labels_example))
        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            (
                sequence_features,
                sentence_features,
            ) = self._extract_labels_precomputed_features(labels_example, attribute)
        else:
            sequence_features = None
            sentence_features = self._compute_default_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL, SEQUENCE, sequence_features)
        label_data.add_features(LABEL, SENTENCE, sentence_features)

        if label_data.does_feature_not_exist(
            LABEL, SENTENCE
        ) and label_data.does_feature_not_exist(LABEL, SEQUENCE):
            raise ValueError(
                "No label features are present. Please check your configuration file."
            )

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )

        label_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

        return label_data

    @staticmethod
    def _compute_default_label_features(
        labels_example: List[Message],
    ) -> List[FeatureArray]:
        """Computes one-hot representation for the labels."""
        logger.debug("No label features found. Computing default label features.")

        eye_matrix = np.eye(len(labels_example), dtype=np.float32)
        # add sequence dimension to one-hot labels
        return [
            FeatureArray(
                np.array([np.expand_dims(a, 0) for a in eye_matrix]),
                number_of_dimensions=3,
            )
        ]

    @staticmethod
    def _find_example_for_label(
        label: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    def _check_labels_features_exist(
        self, labels_example: List[Message], attribute: Text
    ) -> bool:
        """Checks if all labels have features set."""

        return all(
            label_example.features_present(
                attribute, self.component_config[FEATURIZERS]
            )
            for label_example in labels_example
        )
    def _extract_labels_precomputed_features(
        self, label_examples: List[Message], attribute: Text = INTENT
    ) -> Tuple[List[FeatureArray], List[FeatureArray]]:
        """Collects precomputed encodings."""
        features = defaultdict(list)

        for e in label_examples:
            label_features = self._extract_features(e, attribute)
            for feature_key, feature_value in label_features.items():
                features[feature_key].append(feature_value)

        sequence_features = []
        sentence_features = []
        for feature_name, feature_value in features.items():
            if SEQUENCE in feature_name:
                sequence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )
            else:
                sentence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )

        return sequence_features, sentence_features

    def _extract_features(
        self, message: Message, attribute: Text
    ) -> Dict[Text, Union[scipy.sparse.spmatrix, np.ndarray]]:
        (
            sparse_sequence_features,
            sparse_sentence_features,
        ) = message.get_sparse_features(attribute, self.component_config[FEATURIZERS])
        dense_sequence_features, dense_sentence_features = message.get_dense_features(
            attribute, self.component_config[FEATURIZERS]
        )

        if dense_sequence_features is not None and sparse_sequence_features is not None:
            if (
                dense_sequence_features.features.shape[0]
                != sparse_sequence_features.features.shape[0]
            ):
                raise ValueError(
                    f"Sequence dimensions for sparse and dense sequence features "
                    f"don't coincide in '{message.get(TEXT)}'"
                    f"for attribute '{attribute}'."
                )
        if dense_sentence_features is not None and sparse_sentence_features is not None:
            if (
                dense_sentence_features.features.shape[0]
                != sparse_sentence_features.features.shape[0]
            ):
                raise ValueError(
                    f"Sequence dimensions for sparse and dense sentence features "
                    f"don't coincide in '{message.get(TEXT)}'"
                    f"for attribute '{attribute}'."
                )

        # If we don't use the transformer and we don't want to do entity recognition,
        # to speed up training take only the sentence features as feature vector.
        # We would not make use of the sequence anyway in this setup. Carrying over
        # those features to the actual training process takes quite some time.
        if (
            self.component_config[NUM_TRANSFORMER_LAYERS] == 0
            and not self.component_config[ENTITY_RECOGNITION]
            and attribute not in [INTENT, INTENT_RESPONSE_KEY]
        ):
            sparse_sequence_features = None
            dense_sequence_features = None

        out = {}

        if sparse_sentence_features is not None:
            out[f"{SPARSE}_{SENTENCE}"] = sparse_sentence_features.features
        if sparse_sequence_features is not None:
            out[f"{SPARSE}_{SEQUENCE}"] = sparse_sequence_features.features
        if dense_sentence_features is not None:
            out[f"{DENSE}_{SENTENCE}"] = dense_sentence_features.features
        if dense_sequence_features is not None:
            out[f"{DENSE}_{SEQUENCE}"] = dense_sequence_features.features

        return out

