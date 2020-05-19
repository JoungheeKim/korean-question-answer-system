# coding=utf-8
# Kobert is made by SK T-Brain Authors.
# So you may need to contact kobert official gitHub if you need it,
# https://github.com/SKTBrain/KoBERT

from transformers import (
    AutoConfig,
    BertConfig,
    AutoModelForQuestionAnswering,
    BertForQuestionAnswering,
)
import os
from transformers.tokenization_bert import VOCAB_FILES_NAMES
from transformers import PreTrainedTokenizer
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP, PRETRAINED_INIT_CONFIGURATION, PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
import collections
from typing import List, Optional

import logging
from kobert.pytorch_kobert import bert_config
from kobert.utils import download as _download
from kobert.utils import tokenizer as vocab_info
from kobert.pytorch_kobert import pytorch_kobert as model_info
import gluonnlp as nlp
import torch

logger = logging.getLogger(__name__)
cachedir='~/kobert/'


## KOBERT tokenizer 불러오기.
def get_tokenizer(args):
    ## Kobert Tokenizer는 바뀌지 않는다.
    return KoBertTokenizer()

## KOBERT tokenizer 불러오기.
def get_config(args):
    config_path = args.config_name if args.config_name else args.model_name_or_path
    if config_path is None or (not os.path.isdir(config_path) and not os.path.isfile(config_path)):
        config = BertConfig.from_dict(bert_config)
    else:
        config = AutoConfig.from_pretrained(
            config_path,
            cache_dir=args.cache_dir,
        )
    return config


## KOBERT model 불러오기.
def get_model(args, config):
    model = None
    if args.model_name_or_path is None or (not os.path.isdir(args.model_name_or_path) and not os.path.isfile(args.model_name_or_path)):
        model = BertForQuestionAnswering(config)
        # download model
        model_path = _download(model_info['url'],
                               model_info['fname'],
                               model_info['chksum'],
                               cachedir=cachedir)
        model.bert.load_state_dict(torch.load(model_path))
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    return model

class KoBertTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        max_len=512,
        do_lower_case=False,
        **kwargs
    ):
        vocab_path = _download(vocab_info['url'],
                               vocab_info['fname'],
                               vocab_info['chksum'],
                               cachedir=cachedir)
        vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                             padding_token='[PAD]')

        #self.basic_tokenizer = nlp.data.BERTSPTokenizer(vocab_path, vocab_b_obj, lower=do_lower_case)

        ## SentencepieceTokenizer로 변환환
        #self.basic_tokenizer = nlp.data.SentencepieceTokenizer(vocab_path)

        ## 기본 필요 데이터 Extraction
        idx_to_token = vocab_b_obj.idx_to_token
        vocab = {token: idx for idx, token in enumerate(idx_to_token)}

        self.basic_tokenizer = WordpieceTokenizer(vocab, vocab_b_obj.unknown_token)

        super().__init__(
            unk_token=vocab_b_obj.unknown_token,
            sep_token=vocab_b_obj.sep_token,
            pad_token=vocab_b_obj.padding_token,
            cls_token=vocab_b_obj.cls_token,
            mask_token=vocab_b_obj.mask_token,
            **kwargs,
        )
        self.max_len = max_len
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        self.vocab = vocab
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.basic_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        try:
            return self.vocab[token]
        except:
            token = self.unk_token
            return self.vocab[token]

        #return self.basic_tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).replace("▁", " ").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        if token_ids_1 is None, only returns the first portion of the mask (0's).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return KoBertTokenizer()

from transformers.tokenization_bert import whitespace_tokenize
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start == 0:
                        substr = "▁" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens