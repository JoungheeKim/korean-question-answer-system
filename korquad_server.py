# -*- coding: utf-8 -*-
from flask import Flask, render_template
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
from qna_adaptor import QuestionAnswerResponsor
from convert import Korquad2_Converter
from scrap import get_wiki_data
from argparse import ArgumentParser
import time
import sys
import os

app = Flask(__name__)
api = Api(app)
CORS(app)



## 싱글톤 패턴 적용

class SingletonInstane:
    __instance = None

    @classmethod
    def __getInstance(cls, *args, **kargs):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance


class QuestionResponsor(QuestionAnswerResponsor, SingletonInstane):
    pass

class ConvertResponsor(Korquad2_Converter, SingletonInstane):
    pass




class QuestionAnswerServer(Resource):
	def __init__(self):
		super(QuestionAnswerServer, self).__init__()
		self.config = build_parser()
		return

	def getParameter(self, reqparse):
		args = {}
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('question', type=str, default='')
			parser.add_argument('num', type=int, default=1)
			args = parser.parse_args()
			args['question'] = args['question'].strip()
			if type(args['num']) != int or args['num'] == 0:
				args['num'] = 5

		except Exception as e:
			args['question'] = ''
			args['num'] = 5

		return args

	def post(self):
		statusCode = 200
		msg = {}
		try:
			args = self.getParameter(reqparse)

			start_time = time.time()
			_qnaResponsor = QuestionResponsor.instance(self.config.model_name_or_path)
			_convertResponsor = ConvertResponsor.instance()
			print(args['question'])
			print('시간정보1 : ', (time.time() - start_time), 'sec')

			start_time = time.time()
			contents = get_wiki_data(args['question'], int(args['num']))
			print('시간정보2 : ', (time.time()-start_time), 'sec')

			paragraphs = []
			answer = ''

			start_time = time.time()
			for content in contents:
				temp_paragraphs = _convertResponsor.convert_html(content)
				paragraphs.extend(temp_paragraphs)
			print('시간정보3 : ', (time.time() - start_time), 'sec')

			start_time = time.time()
			if len(paragraphs) > 0:
				answer = _qnaResponsor.get_answers(args['question'], paragraphs)
			print('시간정보4 : ', (time.time() - start_time), 'sec')

			msg['answer'] = answer
		except Exception as e:
			statusCode = 500
			msg['answer'] = f'internal error: {str(e)}'

		return msg, statusCode

	def get(self):
		return self.post()



api.add_resource(QuestionAnswerServer, '/qa')

@app.route('/')
def index():
	return render_template('korquad_qna_template.html')


def build_parser():
    parser = ArgumentParser()
    ## BASIC OPTION
    parser.add_argument("--model_name_or_path", dest="model_name_or_path", type=str, required=True, help='학습한 모델이 있는 폴더')
    parser.add_argument("--device", dest="device", default='gpu', type=str, help="모델 구동 환경(GPU/CPU)")
    config = parser.parse_args()
    return config

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=False, use_reloader=False, port=5013)
