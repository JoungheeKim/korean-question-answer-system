# Question & Answer System
이 프로젝트는 고려대학교 강필성 교수님의 [자연어처리](https://github.com/pilsung-kang/text-analytics) 수업에서 진행한 프로젝트입니다. 
  
Q&A 시스템 개발에 가이드라인을 제공하기 위하여 기획한 프로젝트이며 프로젝트에서 사용한 방법론, 데이터, 코드를 다양한 곳에서 참조하여 개발하였습니다.
  
함께한 [DSBA 연구실](http://dsba.korea.ac.kr/) 동료들(이유경, 김혜연, 김명섭) 수고가 많았습니다.
![](imgs/sample_image.gif)

## 프로젝트 진행 영상
프로젝트와 관련된 제안, 중간, 최종 영상입니다.
1. [제안발표 영상](https://youtu.be/JQn5JIthlAI)
2. [중간발표 영상](https://youtu.be/fGQAx_wCm3E)
3. [최종발표 영상]()

## Installation

## Experiment Results
- 총 3개 Model, 4개 Dataset에서 테스트를 진행
- 모든 Hyper-parameter는 고정하고 테스트를 진행
- 평가지표 (EM/F1)
  * EM : Exact Match (%)
  * F1 : F1 Score (%)
- Dataset 구성
  * korquad1.0 + aihub : korquad1.0과 Aihub 데이터를 함께 학습하고, valid 데이터로 korquad1.0 dev를 사용.
  * aihub(8:2) : AIhub 데이터를 8:2나누어 train, valid 데이터로 사용.
- Hyper-parameter Setting
  * max_seq_lenght : 512

|                         | korquad1.0 | korquad1.0 + aihub | aihub (8:2)  | korquad2.0 |
| ----------------------- | --------------- | ------------ | ------------ | ------------ |
| KoBERT                  | 52.81           | 80.27        |              |              |
| Bert-multilingual       | 70.42           | 90.25        |              |              |
| Hanbert                 | 64.32           | 84.78        |              |              |

## Dataset
 - [korquad1.0](https://korquad.github.io/category/1.0_KOR.html)
 - [korquad2.0](https://korquad.github.io/)
 - [Aihub 일반상식](http://www.aihub.or.kr/aidata/84)

## Reference
- [Kobert Github](https://github.com/SKTBrain/KoBERT)
- [HanBert Github](https://github.com/tbai2019/HanBert-54k-N)
- [HanBert Transformer Github](https://github.com/monologg/HanBert-Transformers)
- [KoBERT-KorQuAD Github](https://github.com/monologg/KoBERT-KorQuAD)
- [huggingface transformers Github](https://github.com/huggingface/transformers)

## Source Code Description
- 'korquad_metrics.py', 
