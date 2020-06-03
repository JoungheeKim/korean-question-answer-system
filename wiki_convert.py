from argparse import ArgumentParser
import json
import os
import numpy as np
from tqdm import tqdm
from random import randint
from multiprocessing import Pool
from preprocess import get_wiki_context
from bs4 import BeautifulSoup


def build_parser():
    parser = ArgumentParser()

    ## BASIC OPTION
    parser.add_argument("--worker", dest="worker", default=1, type=int, help='multiprocess 갯수')
    parser.add_argument("--data_path", dest="data_path", default='resource/dev', type=str, help='전환하려는 데이터가 들어있는 폴더')
    parser.add_argument("--save_path", dest="save_path", default='modified/dev', type=str, help='전환후 데이터가 저장될 폴더')
    parser.add_argument("--merge_name", dest="merge_name", default='dev_sample_v2.json', type=str, help='병합파일의 이름')

    ## CONVERT OPTION
    parser.add_argument("--max_paragraph_length", dest="max_paragraph_length", default=768, type=int, help='단락의 길이')  ## 사용안함 .....
    parser.add_argument("--max_answer_text_length", dest="max_answer_text_length", default=128, type=int, help='질문의 길이')  ## 사용안함 .....
    parser.add_argument("--original_qas_include", action="store_true",
                        help='질문에 대한 Original 대답을 포함할 것인지 여부 (TRUE/FALSE)')
    parser.add_argument("--save_each", dest="save_each", action="store_true", default=False,
                        help="각각 전환된 파일을 병합전 저장할지 여부 (TRUE/FALSE)")
    config = parser.parse_args()
    return config


def merge_file(file, merged_files):
    if merged_files is None:
        merged_files = {'version': '', 'data': []}
    merged_files['data'].extend(file['data'])
    return merged_files


def convert_file_into_squad(config):
    if config.merge_name == '':
        config.merge_name = 'temp.json'

    print("INITALIZE Converter")
    ## INIT Converter
    converter = Korquad2_Converter(
        max_paragraph_length=config.max_paragraph_length,
        max_answer_text_length=config.max_answer_text_length
        )

    print("INITALIZE Converter Done..")

    ## get json file list
    file_names = os.listdir(config.data_path)
    file_names = [file_name for file_name in file_names if '.json' in file_name]

    merged_files = None
    if config.worker > 1:

        file_paths = [os.path.join(config.data_path, file_name) for file_name in file_names if '.json' in file_name]
        save_paths = [os.path.join(config.save_path, file_name) if config.save_each else None for file_name in
                      file_names]
        converters = [converter] * len(file_names)

        with Pool(processes=config.worker) as p:
            with tqdm(total=len(file_names), desc='Converting_File', unit='file') as pbar:
                for i, modified_file in enumerate(
                        p.imap_unordered(convert_data_into_squad, zip(file_paths, save_paths, converters))):
                    merged_files = merge_file(modified_file, merged_files)
                    pbar.update()
    else:
        for file_name in tqdm(file_names, desc="Converting File", unit='file'):
            file_path = os.path.join(config.data_path, file_name)
            save_path = None
            if config.save_each:
                save_path = os.path.join(config.save_path, file_name)
            args = file_path, save_path, converter
            modified_file = convert_data_into_squad(args)
            merged_files = merge_file(modified_file, merged_files)
    save_json(os.path.join(config.save_path, config.merge_name), merged_files)


def convert_data_into_squad(args):
    file_path, save_path, converter = args
    json_data = load_json(file_path)
    data_list = json_data['data']
    if converter.worker <= 1:
        data_list = tqdm(data_list)

    modified_data_list = {'version': '', 'data': []}
    for data in data_list:
        modified_paragraphs = converter.convert_to_squad_format(data)
        title = data['title']
        modified_data = {'title': title, 'paragraphs': modified_paragraphs}
        modified_data_list['data'].append(modified_data)

    if save_path:
        save_json(save_path, modified_data_list)
    return modified_data_list


def load_json(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        return data


def save_json(path, data):
    with open(path, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file)


class Korquad2_Converter(object):
    def __init__(self, max_paragraph_length=768, max_answer_text_length=3000):
        self.parser = {
            'table': ['{', '}'],
            'ul': ['{', '}'],
            'dl': ['{', '}'],
            'tr': ['|', '|']
        }

        self.connector = {
            'td': '/',
            'th': '/',
            'dt': ':',
            'dd': ':'
        }
        self.sep_token = '|'

        self.max_paragraph_length = max_paragraph_length
        self.max_answer_text_length = max_answer_text_length

    def get_qas_by_len(self, qas):
        modefied_qas = []
        for qa in qas:
            text = qa['answer']['text']
            text = text.strip()
            if len(text) <= self.max_answer_text_length:
                modefied_qas.append(qa)
        return modefied_qas

    def convert_to_squad_format(self, html, qas):
        structure_contexts = get_wiki_context(html)
        context_list, context_pos_list = self.merge_structure_contexts(structure_contexts)
        paragraphs, paragraph_ids = self.merge_contexts_by_len(context_list, context_pos_list, self.max_paragraph_length)
        assert len(paragraphs) == len(paragraph_ids), "박살났음.. 포기.."
        # print(len(paragraphs), len(paragraph_ids))

        modified_paragraphs = []
        for idx, (paragraph, paragraph_id) in enumerate(zip(paragraphs, paragraph_ids)):
            modified_paragraph = self.get_modified_paragraph(paragraph, paragraph_id, qas)
            if modified_paragraph is not None:
                modified_paragraphs.append(modified_paragraph)
        """
        unique_modified_answers = []
        for modified_paragraph in modified_paragraphs:
            for modified_qa in modified_paragraph['qas']:
                if not modified_qa['is_impossible']:
                    if modified_qa['id'] not in [unique_modified_answer['id'] for unique_modified_answer in unique_modified_answers]:
                        unique_modified_answers.append(modified_qa)
        return modified_paragraphs, unique_modified_answers
        """
        return modified_paragraphs

    ## 테스트용
    def convert_html(self, html):
        structure_contexts = get_wiki_context(html, is_eval=True)
        context_list, context_pos_list = self.merge_structure_contexts(structure_contexts)
        paragraphs, paragraph_ids = self.merge_contexts_by_len(context_list, context_pos_list,
                                                               self.max_paragraph_length)

        return paragraphs

    def get_modified_paragraph(self, paragraph, paragraph_id, qas):
        paragraph_id = np.array(paragraph_id)
        modified_qas = []
        for qa in qas:
            ## 새로 만들 QA
            question = qa['question']
            unique_id = qa['id']
            modified_qa = {"question": question, 'answers': [], 'is_impossible': True, 'id': unique_id}
            answer_start = qa['answer']['answer_start']
            text = qa['answer']['text']
            answer_end = answer_start + len(text) - 1
            possible_ids = np.where((paragraph_id >= answer_start) & (paragraph_id <= answer_end))[0]
            if len(possible_ids) > 0 and possible_ids is not None:
                modified_start = int(min(possible_ids))
                modified_end = int(max(possible_ids)) + 1
                modified_text = paragraph[modified_start:modified_end].rstrip()
                if modified_text is not None and len(modified_text) > 0 and not modified_text == '':
                    modified_answer = {'text': modified_text, 'answer_start': int(modified_start)}
                    modified_qa['answers'].append(modified_answer)
                    modified_qa['is_impossible'] = False
            modified_qas.append(modified_qa)

        if len(modified_qas)>0:
            modified_paragraph = {'context': paragraph, 'qas': modified_qas}
            return modified_paragraph
        return None


    def merge_structure_contexts(self, structure_contexts):
        ## h2, h3로 문단이 나누므로 = 토큰을 사용하여 분리한다.
        h_token = " == "
        h_idx = -1

        paragraphs = []
        paragraph_ids = []

        context_list = []
        context_pos_list = []
        previous_tag = None

        for temp_context in structure_contexts:
            next_tag = temp_context['tag']
            if next_tag in ['h2', 'h3']:
                ## MERGE

                paragraph = self.sep_token.join(context_list)
                paragraph_id = join_list(context_pos_list, self.sep_token)
                paragraphs.append(paragraph)
                paragraph_ids.append(paragraph_id)

                ## INIT
                context_list = []
                context_pos_list = []

            if previous_tag in ['h2', 'h3']:
                context, context_pos, previous_tag = self.merge_structure_text(temp_context, previous_tag, [], [])
                context_list.extend(context)
                context_pos_list.extend(context_pos)

                context_list = [h_token.join(context_list)]
                context_pos_list = [join_list(context_pos_list, h_token)]

            else:
                context, context_pos, previous_tag = self.merge_structure_text(temp_context, previous_tag, [], [])
                context_list.extend(context)
                context_pos_list.extend(context_pos)

        assert len(context_list) == len(context_pos_list), '개망함......'
        if len(context_list) >0 and len(context_pos_list):
            paragraph = self.sep_token.join(context_list)
            paragraph_id = join_list(context_pos_list, self.sep_token)
            paragraphs.append(paragraph)
            paragraph_ids.append(paragraph_id)

        return paragraphs, paragraph_ids


    def merge_structure_text(self, structure_text, previous_tag, context, context_pos):
        text = structure_text['text']
        tag_name = structure_text['tag']
        if type(text) == list and tag_name in ['table', 'ul', 'dl', 'tr']:

            previous_tag = tag_name
            for temp_item in text:
                context, context_pos, previous_tag = self.merge_structure_text(temp_item, previous_tag,
                                                                                        context, context_pos)
            ## MERGE
            context = [self.sep_token.join(context)]
            context_pos = [join_list(context_pos, self.sep_token)]

            return context, context_pos, tag_name

        else:
            text_pos = structure_text['text_pos']

            if tag_name in ['td', 'th'] and previous_tag in ['td', 'th']:
                context[-1] = context[-1] + self.connector[tag_name] + text
                context_pos[-1] = join_list([context_pos[-1], text_pos], self.connector[tag_name])
            elif tag_name in ['dd'] and previous_tag in ['dt']:
                context[-1] = context[-1] + self.connector[tag_name] + text
                context_pos[-1] = join_list([context_pos[-1], text_pos], self.connector[tag_name])
            else:
                context.append(text)
                context_pos.append(text_pos)

            return context, context_pos, tag_name

    def merge_contexts_by_len(self, contexts, context_ids, max_paragraph_length=462):
        assert len(contexts) == len(context_ids), "박살났음...."
        sep_token = '[SEP]'
        sep_idx = -1

        paragraphs, paragraph_ids = [], []

        stack_context = None
        stack_context_id = None
        for context, context_id in zip(contexts, context_ids):
            if stack_context is None and stack_context_id is None:
                stack_context = context
                stack_context_id = context_id
            else:
                if len(stack_context_id) + len(context_id) < max_paragraph_length:
                    stack_context = sep_token.join([stack_context, context])
                    stack_context_id = join_list([stack_context_id, context_id], sep_token)
                else:
                    paragraphs.append(stack_context)
                    paragraph_ids.append(stack_context_id)
                    stack_context = context
                    stack_context_id = context_id
        if stack_context is not None and stack_context_id is not None:
            paragraphs.append(stack_context)
            paragraph_ids.append(stack_context_id)
        return paragraphs, paragraph_ids

def join_list(candidate_list, link_str, idx=-1):
    link_list = [idx] * len(link_str)
    if len(candidate_list) == 0:
        return candidate_list
    if len(candidate_list) == 1:
        return candidate_list[0]

    output_list = []
    if type(candidate_list[0]) == int:
        output_list.append(candidate_list[0])
    else:
        output_list.extend(candidate_list[0])
    for idx in range(1, len(candidate_list)):
        output_list += link_list
        if type(candidate_list[idx]) == int:
            output_list.append(candidate_list[idx])
        else:
            output_list.extend(candidate_list[idx])
    return output_list

if __name__ == '__main__':
    print("###############[START]#################")
    ##load config files
    config = build_parser()
    convert_file_into_squad(config)
    print("###############[END]#################")


