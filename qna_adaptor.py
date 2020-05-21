import datetime
import torch
import os
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.processors.squad import SquadExample, SquadResult
from tqdm import tqdm
import json


from korquad import korquadExample, korquad_convert_examples_to_features
from korquad_metrics import compute_predictions_logits
from kobert_transformers.tokenization_kobert import KoBertTokenizer

from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
)
MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "kobert": (BertConfig, BertForQuestionAnswering, KoBertTokenizer),
    "distilkobert": (DistilBertConfig, DistilBertForQuestionAnswering, KoBertTokenizer),
}
class QuestionAnswerResponsor():
    def __init__(self, model_name_or_path:str='result/', device:str='gpu'):
        ## User Setting
        self.model_name_or_path = model_name_or_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device.lower() in ['gpu', 'cuda'] else "cpu")
        self.batch_size = 1

        ## Training Setting
        training_args = torch.load(os.path.join(model_name_or_path, 'training_args.bin'))
        self.args = training_args

        ##Negative가 모델에 영향을 많이 주기 때문에 정답이 없는 것은 고려하지 않는다.
        self.args.version_2_with_negative = False

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        self.config = config_class.from_pretrained(
            self.model_name_or_path,
            cache_dir=None,
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_name_or_path,
            do_lower_case=self.args.do_lower_case,
            cache_dir=None,
        )
        self.model = model_class.from_pretrained(
            self.model_name_or_path,
            config=self.config,
            cache_dir=None,
        )

        self.model.to(self.device)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )


    def get_answers(self, question:str, paragraphs:list):
        answers = None
        dataset, examples, features = convert_format(question, paragraphs, self.tokenizer, self.args)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.batch_size)
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model

        all_results = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                feature_indices = batch[3]
                outputs = model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                # TODO: i and feature_index are the same number! Simplify by removing enumerate?
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]


                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        temp_dir = 'temp'
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        output_prediction_file = os.path.join(temp_dir, "predictions_.json")
        output_nbest_file = os.path.join(temp_dir, "nbest_predictions_.json")
        output_null_log_odds_file = os.path.join(temp_dir, "null_odds_.json")

        #return all_results
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            self.args.n_best_size,
            self.args.max_answer_length,
            self.args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            self.args.verbose_logging,
            self.args.version_2_with_negative,
            self.args.null_score_diff_threshold,
            self.tokenizer,
        )

        return predictions

def load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return data

class config_parser():
    def __init__(self, **entries):
        self.__dict__.update(entries)


def convert_format(question:str, paragraphs:list, tokenizer, args):
    examples = convert_to_example(question, paragraphs)
    features, dataset = korquad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not True,
        return_dataset="pt",
        threads=args.threads,
    )

    return dataset, examples, features

def convert_to_example(question, paragraphs):

    qas_id = 'text-01'
    question_text = question
    answer_text = None
    title = None
    is_impossible = False

    temp_example = korquadExample(
        qas_id=qas_id,
        question_text=question_text,
        answer_text=answer_text,
        title=title,
        is_impossible=is_impossible,
    )

    for paragraph in paragraphs:
        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            context_text=paragraph,
            answer_text=answer_text,
            start_position_character=None,
            title=title,
            is_impossible=is_impossible,
            answers=[],
        )
        temp_example.add_SquadExample(example)

    return [temp_example]


def to_list(tensor):
    return tensor.detach().cpu().tolist()


