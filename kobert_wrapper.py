# coding=utf-8
# Kobert is made by SK T-Brain Authors.
# So you may need to contact kobert official gitHub if you need it,
# https://github.com/SKTBrain/KoBERT

from transformers import (
    AutoConfig,
    BertConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BertForQuestionAnswering,
)
import os
from transformers.tokenization_bert import VOCAB_FILES_NAMES
from transformers.tokenization_utils import SPECIAL_TOKENS_MAP_FILE, TOKENIZER_CONFIG_FILE

import logging
from kobert.pytorch_kobert import bert_config
from kobert.utils import download as _download
from kobert.utils import tokenizer as vocab_info
from kobert.pytorch_kobert import pytorch_kobert as model_info
import gluonnlp as nlp
import json
import torch

logger = logging.getLogger(__name__)
cachedir='~/kobert/'


## KOBERT tokenizer 불러오기.
def get_tokenizer(args, config):
    tokenizer_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not os.path.isdir(tokenizer_path) and not os.path.isfile(tokenizer_path):
        tokenizer_path = get_kobert_tokenizer(args.max_seq_length)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        config=config,
    )
    return tokenizer

def get_kobert_tokenizer(max_seq_length=512):
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                         padding_token='[PAD]')
    ## 기본 필요 데이터 Extraction
    idx_to_token = vocab_b_obj.idx_to_token
    vocab = {token:idx for idx, token in enumerate(idx_to_token)}
    special_tokens_map = {
        'unk_token':vocab_b_obj.unknown_token,
        'sep_token': vocab_b_obj.sep_token,
        'pad_token': vocab_b_obj.padding_token,
        'cls_token': vocab_b_obj.cls_token,
        'mask_token': vocab_b_obj.mask_token,
    }
    tokenizer_config = {
        'do_lower_case':False,  ## kobert는 lower case를 사용하지 않음.
        'max_len':max_seq_length,
    }

    ## kobert 저장위치
    f_cachedir = os.path.expanduser(cachedir)

    ## kobert Tokenizer 관련 저장위치 설정.
    assert os.path.isdir(f_cachedir), '[{}] kobert 저장위치가 잡히지 않음.'.format(f_cachedir)
    vocab_file = os.path.join(f_cachedir, VOCAB_FILES_NAMES["vocab_file"])
    special_tokens_map_file = os.path.join(f_cachedir, SPECIAL_TOKENS_MAP_FILE)
    tokenizer_config_file = os.path.join(f_cachedir, TOKENIZER_CONFIG_FILE)

    ## kobert / tokenizer_config.json 저장하기.
    with open(tokenizer_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_config, ensure_ascii=False))

    ## kobert / special_tokens_map.json 저장하기.
    with open(special_tokens_map_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(special_tokens_map, ensure_ascii=False))

    ## kobert / vocab.txt 저장하기.
    index = 0
    with open(vocab_file, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(vocab.items(), key=lambda kv: kv[1]):
            if index != token_index:
                logger.warning(
                    "Saving vocabulary to {}: vocabulary indices are not consecutive."
                    " Please check that the vocabulary is not corrupted!".format(vocab_file)
                )
                index = token_index
            writer.write(token + "\n")
            index += 1

    return f_cachedir



## KOBERT tokenizer 불러오기.
def get_config(args):
    config_path = args.config_name if args.config_name else args.model_name_or_path
    if not os.path.isdir(config_path) and not os.path.isfile(config_path):
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
    if not os.path.isdir(args.model_name_or_path) and not os.path.isfile(args.model_name_or_path):
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


