import os
import logging
from transformers import is_torch_available
from transformers.data.processors.squad import whitespace_tokenize, _improve_answer_span, SquadFeatures, _new_check_is_max_context, SquadExample
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np


logger = logging.getLogger(__name__)

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

def korquad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def korquad_convert_examples_to_features(
        examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=korquad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            korquad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        return features, dataset

    return features

def korquad_convert_example_to_features(korquad_example, max_seq_length, doc_stride, max_query_length, is_training):

    features = []
    for example_idx, example in enumerate(korquad_example.get_SquadExamples()):

        if is_training and not example.is_impossible:
            # Get start and end position
            start_position = example.start_position
            end_position = example.end_position

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                return []

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )

        spans = []

        truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
        sequence_added_tokens = (
            tokenizer.max_len - tokenizer.max_len_single_sentence + 1
            if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
            else tokenizer.max_len - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            encoded_dict = tokenizer.encode_plus(
                truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
                span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
                return_token_type_ids=True,
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                if tokenizer.padding_side == "right":
                    non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
                else:
                    last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                    )
                    non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = np.array(span["token_type_ids"])

            p_mask = np.minimum(p_mask, 1)

            if tokenizer.padding_side == "right":
                # Limit positive values to one
                p_mask = 1 - p_mask

            p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

            # Set the CLS index to '0'
            p_mask[cls_index] = 0

            span_is_impossible = example.is_impossible
            start_position = 0
            end_position = 0
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            features.append(
                KorquadFeatures(
                    span["input_ids"],
                    span["attention_mask"],
                    span["token_type_ids"],
                    cls_index,
                    p_mask.tolist(),
                    example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                    unique_id=0,
                    paragraph_len=span["paragraph_len"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    example_idx=example_idx,
                )
            )
    return features

class KorquadFeatures(SquadFeatures):
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        example_idx=None,
    ):
        super().__init__(
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible)
        
        ## 추가
        self.example_idx = example_idx

from transformers.data.processors.squad import SquadProcessor
from convert import Korquad2_Converter
class KorquadV2Processor(SquadProcessor):
    def __init__(self):
        self.converter = Korquad2_Converter()
        super().__init__()

    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            html = entry["context"]
            qas = entry["qas"]

            modified_paragraphs = self.converter.convert_to_squad_format(html, qas)
            temp_examples = {}
            for modified_paragraph in modified_paragraphs:
                context_text = modified_paragraph["context"]

                for qa in modified_paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    if qas_id not in temp_examples:
                        temp_examples[qas_id] = korquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            answer_text=answer_text,
                            title=title,
                            is_impossible=is_impossible,
                        )

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    temp_examples[qas_id].add_SquadExample(example)

            examples.extend([example for qas_id, example in temp_examples.items()])
        return examples

class korquadExample():
    def __init__(
            self,
            qas_id,
            question_text,
            answer_text,
            title,
            is_impossible=False,

    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = []
        self.examples = []

    def add_SquadExample(self, example):
        if not example.is_impossible:
            self.answer_text = example.answer_text
            self.is_impossible = example.is_impossible
            self.answers = example.answers
        self.examples.append(example)

    def get_SquadExamples(self):
        return self.examples
