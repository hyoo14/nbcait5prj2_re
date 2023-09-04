# Relation Extraction o between entities within a sentence

Creating an artificial intelligence model to predict the attributes and relationships of words (entities) in a sentence.

#Boostcamp 5th #NLP

Period| 2023.05.03 ~ 2023.05.18 19:00

[한글로 보기](https://github.com/bootcamphyunwoo/naver_bcait5_lv2_prj2_nlp_klue-re)

## Overview

![](https://lh3.googleusercontent.com/mpLlMKTAwcHmho9058wxavR_PxPrnQyGreQnBuswi0nkbbQQmWCY5OEUwAyZlKWff1Gn9xPHxsVoRXDwv3gky7j5vc8rlg6fvMDvpOTNfe7jeF_zE5GVAVfxd45Ed_b1pJfYqYx2CJzhy_DHGsWfLH0)

문장 속에서 단어간에 관계성을 파악하는 것은 의미나 의도를 해석함에 있어서 많은 도움을 줍니다.

그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.

- 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

이번 대회에서는 문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 모델이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.

{% highlight ruby %}

sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.

subject_entity: 썬 마이크로시스템즈

object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)

{% endhighlight %}  

## Data

- input: sentence, subject_entity, object_entity의 정보를 입력으로 사용 합니다.

- output: relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs을 제출해야 합니다! 클래스별 확률의 순서는 주어진 dictionary의 순서에 맞게 일치시켜 주시기 바랍니다.

## Evaluation Metric

KLUE-RE evaluation metric을 그대로 재현했습니다.

- 1) no_relation class를 제외한 micro F1 score  
  2) 모든 class에 대한 area under the precision-recall curve (AUPRC)

- 2가지 metric으로 평가하며, micro F1 score가 우선시 됩니다.

- Micro F1 score

- micro-precision과 micro-recall의 조화 평균이며, 각 샘플에 동일한 importance를 부여해, 샘플이 많은 클래스에 더 많은 가중치를 부여합니다. 데이터 분포상 많은 부분을 차지하고 있는 no_relation class는 제외하고 F1 score가 계산 됩니다.  
  ![](https://lh4.googleusercontent.com/e_jdTkE69rI6qp5iyjH7cJLahkWKIcX2lH4iQU-bJU_NMCh0f6DBIDFIjE4C-t74SbOBCFPxu3BDrnYwiqbiKPhFX8RJ6vLuNQO3hLfNddKxYTq3SjODPQSxlPeZhOYMHmtOg92EcYq3URSINkjU6so)  
  ![](https://lh6.googleusercontent.com/KMpz5nAck5HC22bVdp_qzLzbva9Bv8UUVz8zrntpF2czd2YCIntuMpBMOiMfGQ_dOHuXiwOfWYqmh_jeJVvvstlSfCP6txeEyHBbQ1mAh9WQbTKt4_Lc8Ng8DFbkHz9eMJmtuZQrLjIr2Uf--s4ngyQ)  
  ![](https://lh6.googleusercontent.com/XE1n3trRxWNCtTDdCbvRmgv4vYYvQ_c5c8vGvFjFRNfnHIadzKO9waOSBAynhQ0u-SmUbwp4w-VtpcFQRutVym12mP57IfZIm8zvJRfiEgIcPYl2PHv_56iHKtG-63ARiUHNbGrhEzn_Ci2j3yU9768)  
  ![](https://lh5.googleusercontent.com/zCFmKivU_oIBA8w7Q2AMAchc4RHv4wgZA0ghjzxlmPKQFMGbQFhHvTuFfGh916wjGneXA9xoTJz3XOw2ssK9oRGJBlEK014MxE6jYfLgIyOlzW4SjZmLT624wlgyydISQIIYI5RK8GGtR2kSnGPbU4s)

- AUPRC

- x축은 Recall, y축은 Precision이며, 모든 class에 대한 평균적인 AUPRC로 계산해 score를 측정 합니다. imbalance한 데이터에 유용한 metric 입니다.  
  ![](https://lh6.googleusercontent.com/gN283psY3GOOoRKtk0mpwCmyG7mZNPVRO3CR6hJh72jPgh-LYqANXNFAKcahk6LY8WenMMLYVR72i983S1Rx3qc3CHx7kmXHOZ4dK6sW3reAzRT-sKTsajjdMsyvcisb_tsqWczk9qvIk3EWQ2T8UM8)

- 위 그래프의 예시는 scikit-learn의 Precision-Recall 그래프의 예시 입니다. 그림의 예시와 같이 class 0, 1, 2의 area(면적 값)을 각각 구한 후, 평균을 계산한 결과를 AUPRC score로 사용합니다.

- Reference

- https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28

- [Precision-Recall &mdash; scikit-learn 1.3.0 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py)

## Detailed Timeline

- 프로젝트 전체 기간 (3주) : 5월 2일 (화) 10:00 ~ 5월 18일 (목) 19:00

- 팀 병합 기간 : 5월 3일 (수) 16:00 까지

- 팀명 컨벤션 : 도메인팀번호(2자리)조 / ex) CV_03조, NLP_02조, RecSys_08조

- 리더보드 제출 오픈 : 5월 3일 (수) 10:00

- 리더보드 제출 마감 : 5월 18일 (목) 19:00

- 최종 리더보드 (Private) 공개 : 5월 18일 (목) 20:00

- GPU 서버 할당 : 5월 2일 (화) 10:00

- GPU 서버 회수 : 5월 19일 (금) 16:00

## Competition Rule

- [대회 참여 제한] NLP 도메인을 수강하고 있는 캠퍼에 한하여 리더보드 제출이 가능합니다.

- [팀 결성 기간] 팀 결성은 대회 페이지 공개 후 2일차 오후 4시까지 필수로 진행해 주세요. 팀이 완전히 결성되기 전까지는 리더보드 제출이 불가합니다.

- [일일 제출횟수] 일일 제출횟수는 '팀 단위 10회'로 제한합니다. (일일횟수 초기화 자정 진행)

- [외부 데이터셋 규정] 본 대회에서는 외부 데이터셋 사용을 금지합니다. 학습에 사용될 수 있는 데이터는 제공되는 train.csv 한 가지 입니다.

- [평가 데이터 활용] test_data.csv에 대한 Pseudo labeling을 금지합니다. test_data.csv을 이용한 TAPT(Task-Adaptive Pretraining)는 허용 합니다. 단 평가 데이터를 눈으로 직접 판별 후 라벨링 하는 행위는 금지합니다. 제공된 학습 데이터을 사용한 데이터 augumentation 기법이나, 생성모델을 활용한 데이터 생성 등, 학습 데이터를 활용한 행위는 허용 됩니다.  
  (학습 효율 측면에서 테스트셋의 라벨링을 추론하는 행위는 본 대회에서는 금지합니다)

- [데이터셋 저작권] 대회 데이터셋은 '캠프 교육용 라이선스' 아래 사용 가능합니다. 저작권 관련 세부 내용은 부스트코스 공지사항을 반드시 참고 해주세요.

---

AI Stages 대회 공통사항

- [Private Sharing 금지] 비공개적으로 다른 팀과 코드 혹은 데이터를 공유하는 것은 허용하지 않습니다.  
  코드 공유는 반드시 대회 게시판을 통해 공개적으로 진행되어야 합니다.

- [최종 결과 검증 절차] 리더보드 상위권 대상으로추후 코드 검수가 필요한 대상으로 판단될 경우 개별 연락을 통해 추가 검수 절차를 안내드릴 수 있습니다. 반드시 결과가 재현될 수 있도록 최종 코드를 정리 부탁드립니다. 부정행위가 의심될 경우에는 결과 재현을 요구할 수 있으며, 재현이 어려울 경우 리더보드 순위표에서 제외될 수 있습니다.

- [공유 문화] 공개적으로 토론 게시판을 통해 모델링에 대한 아이디어 혹은 작성한 코드를 공유하실 것을 권장 드립니다. 공유 문화를 통해서 더욱 뛰어난 모델을 대회 참가자 분들과 같이 개발해 보시길 바랍니다.

- [대회 참가 기본 매너] 좋은 대회 문화 정착을 위해 아래 명시된 행위는 지양합니다.

- 대회 종료를 앞두고 (3일 전) 높은 점수를 얻을 수 있는 전체 코드를 공유하는 행위

- 타 참가자와 토론이 아닌 단순 솔루션을 캐내는 행위

### Statistics of Dataset

---

- 전체 데이터에 대한 통계는 다음과 같습니다.

- train.csv: 총 32470개

- test_data.csv: 총 7765개 (정답 라벨 blind = 100으로 임의 표현)

학습을 위한 데이터는 총 32470개 이며, 7765개의 test 데이터를 통해 리더보드 순위를 갱신합니다. public과 private 2가지 종류의 리더보드가 운영됩니다.

- dict_label_to_num.pkl: 문자 label과 숫자 label로 표현된 dictionary, 총 30개 classes (class는 아래와 같이 정의 되어 있며, 평가를 위해 일치 시켜주시길 바랍니다.) pickle로 load하게 되면, 딕셔너리 형태의 정보를 얻을 수 있습니다.

with open('./dict_label_to_num.pkl', 'rb') as f:

    label_type = pickle.load(f)

{'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, 'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, 'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, 'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, 'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}

dict_num_to_label.pkl: 숫자 label과 문자 label로 표현된 dictionary, 총 30개 classes (class는 아래와 같이 정의 되어 있며, 평가를 위해 일치 시켜주시길 바랍니다.) pickle로 load하게 되면, 딕셔너리 형태의 정보를 얻을 수 있습니다.

with open('./dict_num_to_label.pkl', 'rb') as f:

    label_type = pickle.load(f)

{0: 'no_relation', 1: 'org:top_members/employees', 2: 'org:members', 3: 'org:product', 4: 'per:title', 5: 'org:alternate_names', 6: 'per:employee_of', 7: 'org:place_of_headquarters', 8: 'per:product', 9: 'org:number_of_employees/members', 10: 'per:children', 11: 'per:place_of_residence', 12: 'per:alternate_names', 13: 'per:other_family', 14: 'per:colleagues', 15: 'per:origin', 16: 'per:siblings', 17: 'per:spouse', 18: 'org:founded', 19: 'org:political/religious_affiliation', 20: 'org:member_of', 21: 'per:parents', 22: 'org:dissolved', 23: 'per:schools_attended', 24: 'per:date_of_death', 25: 'per:date_of_birth', 26: 'per:place_of_birth', 27: 'per:place_of_death', 28: 'org:founded_by', 29: 'per:religion'}

### Data Example

---

### ![](https://lh5.googleusercontent.com/eMkMtiLCj5wmKOTEcbUxAbnJ2OuTxhwoOb59Z3eSzTJ5I1-ipew1WfZVKhxaOs3moZ0jZHHkkJQcuVGRm9agDnQFjazkgjSm1KypoBd-Ynvrs3dAZPpiiwSOlVhnbaZuzuMdeUJ123nVk_HGZTgJwFU)

- column 1: 샘플 순서 id

- column 2: sentence.

- column 3: subject_entity

- column 4: object_entity

- column 5: label

- column 6: 샘플 출처

- class에 대한 정보는 2개의 dictionary를 따라 주시기 바랍니다.

![](https://lh5.googleusercontent.com/Xnfar2SnjwLE3XtwSVzOFPRvKJQT5VTk8w4Ut9q1LOszZz2lpTN6VGLKuzdhj4LtbZYePsC8r44-OR4KytIM9vWCw9vGPWv3EG_jTE3pwvI6iv2d57XmRVgY0u-lpSIAQnLz8e3UcbtlUpWCUh6OL1U)

### Detailed Competition Rule

---

- [외부 데이터셋 규정] 외부 데이터 사용은 금지합니다. 학습에 사용될 수 있는 데이터는 AI Stages 에서 제공되는 데이터셋 한 가지 입니다.

- [평가 데이터 활용] test_data.csv에 대한 Pseudo labeling을 금지합니다. test_data.csv을 이용한 TAPT(Task-Adaptive Pretraining)는 허용 합니다. 단 평가 데이터를 눈으로 직접 판별 후 라벨링 하는 행위는 금지합니다. 제공된 학습 데이터을 사용한 데이터 augumentation 기법이나, 생성모델을 활용한 데이터 생성 등, 학습 데이터를 활용한 행위는 허용 됩니다.  
  (학습 효율 측면에서 테스트셋의 라벨링을 추론하는 행위는 본 대회에서는 금지합니다)

- 테스트 데이터셋에 대한 통계는 다음과 같습니다.

- test_data.csv: 총 7765개 (정답 라벨 blind = 100으로 임의 표현)

- 이렇게 주어진 7765개의 test 데이터를 통해 리더보드 순위를 갱신합니다.

- Public, Private 데이터는 각각 50% 비율로 무작위로 선정되었습니다.

- Public (대회 진행중)

- test_data.csv로 만든 submission.csv 파일을 통해 자동으로 public과 관련된 샘플들을 평가하게 됩니다.

- Private (대회 종료후)

- test_data.csv로 만든 submission.csv 파일을 통해 자동으로 private과 관련된 샘플들을 평가하게 됩니다.

- train.csv는 학습을 위한 label 정보가 포함되어 있는 반면, test_data.csv는 label에 대한 정보들은 blind를 위해 임의로 100이라는 공통 label로 되어 있습니다.

Baseline 디렉토리 구조  
├── code

│   ├── __pycache__

│   ├── best_model

│   ├── logs

│   ├── prediction

│   └── results

└── dataset

    ├── test

-     └── train

Baseline 파일 포함 디렉토리 구조  
├── code

│   ├── __pycache__

│   │   └── load_data.cpython-38.pyc

│   ├── best_model

│   ├── dict_label_to_num.pkl

│   ├── dict_num_to_label.pkl

│   ├── inference.py

│   ├── load_data.py

│   ├── logs

│   ├── prediction

│   │   └── sample_submission.csv

│   ├── requirements.txt

│   ├── results

│   └── train.py

└── dataset

    ├── test

    │   └── test_data.csv

    └── train

-         └── train.csv

- code 설명

- [train.py](http://train.py)

- baseline code를 학습시키기 위한 파일입니다.

- 저장된 model관련 파일은 results 폴더에 있습니다.

- [inference.py](http://inference.py)

- 학습된 model을 통해 prediction하며, 예측한 결과를 csv 파일로 저장해줍니다.

- 저장된 파일은 prediction 폴더에 있습니다.

- sample_submission.csv를 참고해 같은 형태로 제출용 csv를 만들어 주시기 바랍니다.

- load_data.py

- baseline code의 전처리와 데이터셋 구성을 위한 함수들이 있는 코드입니다.

- logs

- 텐서보드 로그가 담기는 폴더 입니다.

- prediction

- [inference.py](http://inference.py) 를 통해 model이 예측한 정답 submission.csv 파일이 저장되는 폴더 입니다.

- results

- train.py를 통해 설정된 step 마다 model이 저장되는 폴더 입니다.

- best_model

- 학습중 evaluation이 best인 model이 저장 됩니다.

- dict_label_to_num.pkl

- 문자로 되어 있는 label을 숫자로 변환 시킬 dictionary 정보가 저장되어 있습니다.

- dict_num_to_label.pkl

- 숫자로 되어 있는 label을 원본 문자로 변환 시킬 dictionary 정보가 저장되어 있습니다.

- custom code를 만드실 때도, 위 2개의 dictionary를 참고해 class 순서를 지켜주시기 바랍니다.

- dataset 설명.

- train

- train 폴더의 train.csv 를 사용해 학습 시켜 주세요.

- evaluation data가 따로 제공되지 않습니다. 적절히 train data에서 나눠 사용하시기 바랍니다.

- test

- test_data.csv를 사용해 submission.csv 파일을 생성해 주시기 바랍니다.

- 만들어진 submission.csv 파일을 리더보드에 제출하세요.

베이스라인 코드 설명

- 이번 competition의 베이스라인의 코드는 huggingface를 사용한 klue/bert-base model을 바탕으로 작성 되었습니다.

- 베이스라인에 사용한 전처리된 Dataset sample 입니다. sentence, subject_entity, object_entity를 model의 input으로 입력 받아, 30개의 classes 중 1개를 예측하도록 합니다.![](https://lh4.googleusercontent.com/5egMdva6FGv3A6FPVAQh--OtcJyyEZMJzC0Wok3r8y-At1LYK2eyIyBx6iGaKIbd0WyvbImidZizcuACD2yO013v2j_IVC8VrnAl4ft1YBS_JtuzeOfc9C4pxCD3jprStsoyPx7hvsXApMef0m3zJG8)

- 간단하게 subject_entity, object_entity, sentence를 special token인 [SEP]을 활용해 분리하고, model의 input으로 사용했습니다.

- ※ 주어진 train.csv 파일은 의도적으로 label을 포함하지 않은 raw dataset으로 제공합니다. baseline에서 벗어나 자유로운 시도를 하고 싶다면, 경우에 따라 2개의 dictionary를 사용해 직접 전처리 작업을 하셔야 합니다.

- ex) [CLS] subject_entity [SEP] object_entity [SEP] sentence [SEP]

['[CLS]', 'UN', '##M', '[SEP]', '미국', '[SEP]', '세계', '대학', '학', '##술', '순', '##위', '(', 'Academic', 'Ranking', 'of', 'World', 'Universities', ')', '에서', '##는', 'UN', '##M', '##을', '세계', '##순', '##위', '201', '-', '300', '##위로', '발', '##표', '##했고', '미국', '내', '순', '##위로', '##는', '90', '-', '110', '##위로', '발', '##표', '##했다', '.', '[SEP]',]

1. 라이브러리 설치
- 압축을 푼 후 아래 명령어를 실행해 필요한 라이브러리를 설치해 줍니다.

pip install -r requirements.txt

2. Training
- 기본으로 설정된 hyperparameter로 [train.py](http://train.py) 실행합니다.

- baseline code에서는 500 step마다 logs 폴더와 results 폴더에 각각 텐서보드 기록과 model이 저장됩니다.

python train.py

3. Inference
- 학습된 모델을 추론합니다. default는 ./best_model 경로로 되어 있습니다. 저장된 모델의 checkpoint에 맞게 변경해서 inference 하시기 바랍니다.

- results 폴더에 저장된 checkpoint 사용시  
  python inference.py --model_dir=./results/checkpoint-500

- 제출을 위한 csv 파일을 만들고 싶은 model의 경로를 model_dir에 입력해 줍니다.

- 오류 없이 진행 되었다면, ./prediction/submission.csv 파일이 생성 됩니다.

- 생성된 파일을 제출해 주시기 바랍니다.

- baseline model을 기준으로 작성되어 있습니다! model이 변경되었을 경우, inference.py에서 알맞은 model로 변경하시기 바랍니다.

python inference.py --model_dir=./best_model

- 제출할 csv 파일은 반드시 column명이 id, pred_label, probs로 되어야 하며, id는 test_data의 순서와 같고, pred_label은 문자로된 실제 class , probs는 30개 class에 대한 확률 (logit 값이 아닌 전체 class확률의 합이 1인 확률들을 의미 합니다.) 입니다.

- code/prediction/sample_submission.csv를 참고해주세요.

![](https://lh4.googleusercontent.com/RCy6d-IEbUowv7IvKMTacUyc2tH6OkgYkFJ_sILc2JR9_uY-mwlJ5V3geFQ0CxzN1VW8tty38iEuP6H6ut-h7pEeZiKQiWJh6EtRYjf7D6kCOoHKqWsYHIVkkOEOL9heVsqCuFCzLSfpMySAa9BdYyI)

### Download Links

Download Data Link

https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000239/data/data.tar.gz

Download Baseline Code Link

https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000239/data/code.tar.gz

### Leader Board

![](https://lh3.googleusercontent.com/7ooTJiYuNn1JiWUgpM6h_KszdFm4YBSFtzHLkyHqys3-MI9qacwJcIb5UsrTxXT2DP8aF59sKMemaGZJDofiSQzqSf9exRR2e4y8BqanFeS7S9Bsc2_9VsgCt68a3J9pF7FPjDYrARppJJ0V3tu_cfE)

![](https://lh4.googleusercontent.com/LO-pdKBNuIBTBSfvePMmwigX8AdW6TKQu4sVtBt8m9NKbXJPqgi9SlEYieX7HF3GyfHjccJ65Mtg4976F1TlVOdLtBuwWMDL8Iudjd8TRUYHMDANjZc5KZVhjSzCLdFWwEyRrw7UoDez7WmoPACbHjQ)

## ETC

[공유] 딥러닝 모델의 추론 결과를 어떻게 분석하면 좋을까?

Posted by 이녕우_조교

2023.04.24.10:27

 

학습 데이터로 딥러닝 모델을 학습했다면, 학습한 모델이 어느 정도의 성능을 내는지 측정할 수 있어야 합니다. 여기서 모델은 이전에 본 적이 없는 데이터에도 일반화(Generalization)가 잘 되어있어야 하므로, 우리는 보통 모델이 학습하지 않은 데이터들로 평가 데이터셋을 구성하여 모델을 평가하게 됩니다.

현업에서는 좋은 성능을 가지는 모델을 얻기 위해서, (1) 모델을 학습하고, (2) 학습된 모델의 추론 결과를 평가하고, (3) 평가 결과를 분석하여 문제점을 찾아 다시 모델을 학습하는 일련의 싸이클을 여러 번 거칩니다. 개인적으로, 여기서 가장 중요한 과정은 모델의 평가 결과를 분석하고 문제점을 찾는 과정이라고 생각해요. 하지만, 이러한 싸이클을 많이 겪어보지 않은 분들은 학습한 모델의 추론 결과에 대한 분석을 소홀히하거나, 문제점을 면밀하게 확인하지 않은 채, 단순히 성능 향상을 위한 golden 솔루션을 찾는 경우가 많습니다. 그러다보면 내가 시도한 솔루션(원인)과 모델의 성능(결과)에 대한 인과 관계를 찾기 어렵게 되고, 결과적으로 무작정 하이퍼파라미터 서칭만 하게 되는 경우가 많은 것 같아요.

그래서 저는 딥러닝 모델을 학습한 후, 학습한 모델의 추론 결과를 어떻게 분석하면 좋을지 간단하게 얘기해보고 싶습니다.

먼저, 모델의 성능을 적절하게 평가하려면 작업(Task)의 목표에 맞는 적절한 성능 평가 지표를 선택해야합니다. 예를 들어서, KLUE 벤치마크 내 Relation Extraction 데이터셋의 경우에는 Micro F1 score를 성능 평가 지표로 사용하고 있어요.

![](https://lh4.googleusercontent.com/ORXAY5ouW2WQ68awMtOJoEMbHEHXZ03Zf9Pm-iPgfwy-Ojm8CAvJtg72-dq4D6qaAl8eJu4J0wK6JtXNoR1Pf1R_-UHJDOVh4PaYmR3LzKIhs6X8Ldifj4mW35_5QHkrz0pT1UGgjZnoMEu0mRhFhB0)

간단하게는, 이 지표를 보고 모델을 어떻게 분석하면 좋을지에 대한 힌트들을 얻을 수 있는데요. 가장 먼저 확인할 수 있는 것은, 일련의 수식에 의해 얻어진 F1 score입니다. 아마 대부분의 경우에, 많은 분들이 이 F1 score가 올랐는지/내렸는지를 보고 실험 결과를 판단하고, 이후 실험들에 대한 결정들을 하게될 것 같아요.

그리고, 우리는 추가로 precision, recall 성능들을 측정해볼 수 있어요. 수식에 따르면, precision과 recall 성능을 올리면 F1 score가 높아질 수 있거든요.

더 나아가서, 우리는 TP(True Positive), FP(False Positive), FN(False Negative)등과 같은 보다 세부적인 지표에 집중할 수도 있습니다. TP를 높이거나, FP나 FN을 낮추면 precision, recall 성능이 오르고, 결과적으로 F1 score가 높아질 수 있거든요. 특히, 이러한 지표들은 단순히 정량적인 수치 확인 뿐만 아니라, 각 항목에 대한 데이터를 정성적으로 확인해볼 수 있다는 장점이 있습니다.

특히, FP(False Positive), FN(False Negative)에 속하는 데이터들을 정성적으로 확인하는 것은, 학습한 모델이 지금 어떤 데이터들을 헷갈려하는지 확인할 수 있다는 점에서 성능 향상에 매우 유용한 힌트가 될 수 있어요.

이런 지표를 한 눈에 볼 수 있도록, 우리는 Confusion Matrix를 활용하곤 합니다.

![](https://lh3.googleusercontent.com/-A8RHm4uuemDm1fD0TfjWdHiTUbzFn-L_U6JG1n6fcESQb7zPeY3G7T5lRNiN4YqSovvMJExgDJ_E0j48tvLxJMOiMnpHjMRpoLYmas6zJwnQVMPZ9mN4cysS-jiWF77wDKcfQG3nQWrHe0ueEXPB9I)

모델이 예측한 데이터와, 정답 레이블을 가지고 Confusion Matrix를 그리는 간단한 방법을 공유할게요.

관련 내용은 다음의[포스트](https://jerimo.github.io/python/confusion-matrix/)를 참조했습니다.

from sklearn.metrics

import confusion_matrix, plot_confusion_matrix

import matplotlib.pyplot as plt

label=['anger', 'happiness', 'fear', 'sadness', 'neutral'] # 라벨 설정

plot = plot_confusion_matrix(clf, # 분류 모델

                             y_predicted, y_true, # 예측 데이터와 예측값의 정답(y_true)

                             display_labels=label, # 표에 표시할 labels

                             cmap=plt.cm.Blue, # 컬러맵(plt.cm.Reds, plt.cm.rainbow 등이 있음)

                             normalize=None) # 'true', 'pred', 'all' 중에서 지정 가능. default=None

plot.ax_.set_title('Confusion Matrix')

위 코드의 실행 결과는 다음과 같습니다. (left: normalized=None, right: normalized=’true’)

![](https://lh5.googleusercontent.com/HkmBBwOhIXeX-uZhbnRwLLI9NsBzMsY3v6Q6McmJdrPzDLs3RxT8ZKlYiFcKu_GoZHxOlcMK8zas0Uvc02CDG0BSWtBvwlHAFuedw52m5dCaFzYg2MjsNPb00AvZf3HTmvSRRBsHzr0pH2JDddY4G5c)  
위 confusion matrix를 보면, 예시 모델이 anger와 fear를, happiness와 neutral을 조금 더 헷갈리고 있음을 확인할 수 있고, 각 cell에 해당하는 데이터들을 정성적으로 보면 모델이 왜 헷갈려하는지, 그렇다면 이것을 어떻게 해결할 수 있는지에 대한 힌트를 얻을 수 있습니다.

[공유] 데이터 편향 확인을 위한 데이터셋 EDA

Posted by 이녕우_조교

2023.04.24.10:28

일반적으로 딥러닝 모델 학습에 사용하는 학습 데이터는 현실의 일부를 다루기 때문에 현실의 분포를 완전히 설명하지 못합니다. 이러한 학습 데이터를 학습하는 딥러닝 모델은 필연적으로 편향성을 가지게 되는데요. 여기서 말하는 편향성이란, 딥러닝 모델이 특정 결과에 대해 어느 정도의 편향성을 가지고 판단을 내리거나, 데이터 내 어떤 특징들을 이해하는데 데이터의 부분집합에 의존하고 있다는 말입니다.

한 예시로, 대형 언어 모델인 GPT-3 ([논문 링크](https://arxiv.org/abs/2005.14165))가 가지는 무슬림에 대한 편향을 들 수 있는데, 이를 보고하는[논문](https://dl.acm.org/doi/abs/10.1145/3461702.3462624)은 무슬림과 폭력성에 대한 연관성이 모델의 다양한 사용 사례에서 일관되고 창의적으로 나타난다는 것을 발견했습니다.

![](https://lh3.googleusercontent.com/6xA2JnNIQq4UauNXzJFHevue4bsMcxDiHBAKQHSeQ6z5RcC9SRAegfTPuWhzOCfRffjt6yKIHhtjAFbgZSAVOKz5er5RmCA7wT4PjJ4bXkhWnsALtujVUsfA8izfihhH-Fw5MwH8N-eB6vAnvsB-d_Q)

“무슬림과 폭력 사이의 이러한 연관성은 학습되지만 암기되지 않는 것 같습니다. 오히려 GPT-3은 기본 편향을 매우 창의적으로 나타내어 언어 모델이 편향을 다양한 방식으로 변형할 수 있는 강력한 능력을 보여주므로 편향을 감지하고 완화하기가 더 어려울 수 있습니다."

이러한 현상이 발생한 원인을 생각해보면, GPT-3가 학습한 학습 데이터셋 대부분이 인터넷에 공개된 데이터였고, 인터넷에 무슬림과 관련한 편향적인 글들이 많이 있었기 때문에, 이를 학습한 모델이 자연스레 해당 편향을 학습했을거라 추측할 수 있습니다.

이처럼 모델의 편향은 학습 데이터셋의 편향에서 기인하는데요. 그렇기 때문에 우리는 이후 모델이 가질 편향을 조금이나마 예측하고 대응하기 위해서, 데이터셋에 어떤 편향이 존재하고 있는지 조사해야할 필요가 있습니다. 데이터 편향에 대해 궁금한 분들은 다음 자료를 읽어보면 좋을 것 같아요! → [Understanding Data Bias](https://towardsdatascience.com/survey-d4f168791e57)

대회 참가자의 관점에서, 만약 내가 가진 학습 데이터셋에 편향이 있고 반면 테스트셋은 상대적으로 공정하다면, 편향된 학습 데이터셋으로 학습한 모델은 테스트셋 성능에서 손해를 보게 될 수 있습니다. 따라서, 학습 데이터셋 내 편향을 검사하고, 발견된 편향이 성능에 미치는 영향에 대해 생각해봐야 합니다.

Relation Extraction (RE)과 같은 데이터셋에도 당연히 편향이 존재할 수 있습니다. KLUE 벤치마크의 Relation Extraction 데이터셋을 예시로 들면, 이 데이터셋에는 총 30가지의 관계(Relation) 클래스가 존재하며, 아래 표에서 볼 수 있듯 클래스 간 데이터 불균형이 존재하는 것을 확인할 수 있습니다. 이러한 클래스 간 불균형 문제는 데이터셋 EDA를 진행할 때 기본적으로 확인해야하는 통계입니다.

https://klue-benchmark.com/tasks/70/data/description

![](https://lh4.googleusercontent.com/wwsbo1Jh3NZRaLBcEO2stE2ST53sdRqlEZrJ-6ZPqYRKydwlcMjKakWS6nByJ74d_WH_hXgJXyR_iuWgTsFXFZUF7HUO-gYA2uzrk2taCabZcW3TMnKP0fo74n6xJ3ujObagSJ72_Au6hpYnbBcNtPk)  

간단하게, 우리는 특정 클래스와 같이 자주 나오는 단어들을 WordCloud를 통해 확인하여, 관계 클래스와 단어 간의 의도하지 않은 편향들이 있는지 확인해볼 수 있습니다. 이번 예시에서는 per:employeeof class와 subjectentity 간의 상관관계를 알아보고자 합니다.

{% highlight ruby %}

from wordcloud import WordCloud

from konlpy.tag import Twitter

from collections import Counter

#학습 데이터셋를 베이스라인 코드의 load_data 함수를 사용하여 읽어옵니다.

traindataset = loaddata(dataset_path)

employeeoflist = []for index, data in train_dataset.iterrows():

  if data["label"] == "per:employee_of":

    employeeoflist.append(data["subject_entity"])

#가장 많이 나온 단어부터 40개를 저장한다.

counts = Counter(employeeoflist)

tags = counts.most_common(40)

#WordCloud를 생성한다.

wc = WordCloud(fontpath = path, backgroundcolor="white", maxfontsize=60)

cloud = wc.generatefromfrequencies(dict(tags))

#생성된 WordCloud를 출력한다.

plt.figure(figsize=(10, 8))

plt.axis('off')

plt.imshow(cloud)

plt.show()

{% endhighlight %}  

생성된 word cloud를 출력하면 다음과 같습니다.![](https://lh5.googleusercontent.com/Lan1rUqI3-_UF2uy-NHgA8KvJ3pK16QFNqzOXLLdQ1wEbY8ms1wbTqs_StKevOzWhExcm5vRctKrfokuOH68JsSsxH-IlnsjpLnLx47OT7nZK-vyH5KiBH1xaDVLMoDPefOnNKvmO2pAF_YaGRF2hEU)

간단한 예시이나, 우리는 위 word cloud를 통해 per:employeeof 관계 class의 subjectentity가 대부분 이름이라는 것을 확인할 수 있습니다. 따라서, 우리는 subjectentity가 이름인 경우에, per:employeeof class와 의도하지 않은 의존 관계가 생길 수 있음을 의심해볼 수 있습니다.

이후에도, 모델 추론 결과를 정성적으로 확인할 때 이렇게 데이터셋에서 얻은 인사이트가 모델의 잘못된 판단을 해석하는데 도움을 줄 수 있으므로 학습 데이터셋을 면밀히 살펴보면 좋을 것 같아요. 각자의 데이터셋 분석 결과를 댓글이나 게시글을 통해 공유해보면 좋을 것 같아요!

Commented by 문지혜_T5077

2023.05.06.18:08

!pip install wordcloud 로 패키지를 설치하고 위의 코드를 실행하면, AttributeError: 'TransposedFont' object has no attribute 'getbbox' 에러가 발생합니다. 해당 에러는 패키지를 1.9.1.1 로 버전 업(2023년 4월 28일) 하며 발생하는 이슈인 것으로 판단됩니다.

!pip install wordcloud==1.8.2.2 이전 버전으로 설치하니 이녕우 조교님이 게시물에 작성하신 코드가 에러 없이 잘 수행되었습니다. :)

[공유] 학습된 BERT classifier의 attention map 그려보기

Posted by 이녕우_조교

2023.04.24.10:29

 

최근 사용되는 BERT, RoBERTa와 같은 사전학습 모델들은 대부분 Transformer 기반의 구조를 가지고 있습니다. 그리고 Transformer는 self-attention mechanism을 통해 입력된 문장의 의미를 이해합니다. BERT가 입력 문장 내의 어떠한 단어들에 집중하였는지 살펴보기 위해 Heatmap을 그려 self-attention score를 시각화해볼 수 있습니다.

먼저 아래와 같이 문장의 감성(sentiment)를 분류하도록 학습된 BERT 모델을 불러와 inference를 진행합니다.

{% highlight ruby %}

from transformers import AutoModel, AutoTokenizer

import torch

modelname = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(modelname)

classifier = AutoModel.from_pretrained(modelname)

sentence = 'I like this road.'

encoded = tokenizer(sentence,return_tensors='pt')

print(encoded)

#{'input_ids': tensor([[ 101, 1045, 2066, 2023, 2346, 1012,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}

output = classifier(encoded['input_ids'],encoded['attention_mask'], output_attentions=True, return_dict=True)

attention = output.attentions

#attention의 모양은 6 * [1, 12, 7, 7] num_layer * (batch_size, num_attention_head, sequence_length,sequence_length)입니다.

attention = torch.stack(attention).squeeze().detach().cpu().numpy()

#(num_layer, num_attention_head, sequence_length,sequence_length)

tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

print(attention.shape) # (6, 12, 7, 7)

{% endhighlight %}  

위의 예시와 같이, Huggingface 모델의 forward 과정에서 output_attentions=True 를 입력하면 모델의 예측 결과와 함께 각 layer 및 head의 self-attention probability가 반환됩니다.

이제 모델의 각 layer에 있는 여러 head들에서 단어들 사이의 attention score를 시각화해 봅시다.

여기서 사용한 distillbert 모델은 총 6개의 layer로 이루어져 있고, 하나의 layer에는 12개의 head가 포함되어 있습니다. 따라서 한 layer에 총 12개의 heatmap을 그릴 수 있습니다.

{% highlight ruby %}

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

target_layer = 0

fig = plt.figure(figsize=(15,15))

fontdict = {'fontsize': 8}

num_token = len(tokens)

for head_idx, head_score in enumerate(attention[target_layer]):

    ax = fig.add_subplot(4, 3, head_idx+1)

    img = ax.imshow(np.array(head_score))

    ax.set_xticks(range(num_token))

    ax.set_yticks(range(num_token))

    ax.set_xticklabels(tokens, fontdict=fontdict, rotation=90)

    ax.set_yticklabels(tokens, fontdict=fontdict)

    ax.set_xlabel('HEAD'+str(head_idx))

    fig.colorbar(img)

plt.tight_layout()

plt.show()

{% endhighlight %}  

위 코드를 실행시키면 아래와 같은 시각화 결과를 확인할 수 있습니다. 그런데 BERT, GPT와 같은 사전학습 모델의 경우 layer와 attention head의 개수가 많기 때문에 한눈에 attention score를 보기 어려울 때가 있습니다. 이 경우, 한 layer에 있는 여러 head에서 나온 attention score의 평균을 구해서 시각화하거나, 모델 전체의 layer와 head에서 나온 attention score의 평균을 구해서 하나의 그림을 그릴 수도 있습니다.

[공유] Tokenized sequence에서 entity의 위치 찾기

Posted by 이녕우_조교

2023.04.24.10:29

 

Named Entity Recognition (NER), Relation Extraction (RE)과 같은 token classification 관련 task의 데이터셋에서는 string index를 활용하여 우리가 관심이 있는 객체(entity)의 위치를 표시하는 경우가 많습니다.

예를 들어 KLUE 벤치마크의 Relation Extraction 데이터셋을 살펴보면 아래와 같습니다.

sentence: "〈Something〉는 **조지 해리슨**이 쓰고 **비틀즈**가 1969년 앨범 《Abbey Road》에 담은 노래다."

subject_entity: { "word": "비틀즈", "start_idx": 24, "end_idx": 26, "type": "ORG" }

object_entity: { "word": "조지 해리슨", "start_idx": 13, "end_idx": 18, "type": "PER" }

이때 우리가 관심있는 객체의 string(e.g, “비틀즈”) 뿐만 아니라 해당 객체의 시작 및 끝 위치(24, 26)까지 함께 데이터셋에서 제공하는 이유는, 입력 문장에서 해당 객체가 여러 번 등장했을 때 이 중 어떤 등장이 우리가 관심있는 대상인지 알 수 없기 때문입니다.

예를 들어 “이순신 장군은 조선 중기의 무신이며, 이순신 장군의 동상은 대한민국에 있다.”라는 문장이 있을 때, “이순신”이라는 entity의 (startidx, endidx)는 (0,2)이 될 수도 있고 (21, 23)이 될 수도 있습니다.

그런데 이러한 token-classification 데이터셋 내의 문장을 tokenization할 경우, 일반적인 tokenizer는 entity의 startidx와 endidx를 고려하여 tokenization을 수행하지 않기 때문에 tokenization이 끝난 token들과 데이터셋에서 주어진 entity를 연결하는 과정이 어려운 경우가 많습니다.

이와 같은 불편함을 해소하기 위해 Huggingface Tokenizer에서는 tokenization이 끝난 token들이 원래 문장에서 어떤 string index를 가졌는지 알려주는 기능을 제공합니다. 아래와 같이 tokenizer의 forward 함수에 returnoffsetsmapping=True를 함께 넣어주면 각 토큰들의 원본 입력에서의 string offset이 “offset_mapping” key의 value로써 함께 반환됩니다.

{% highlight ruby %}

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

sentence = "〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다."

output = tokenizer(sentence, return_offsets_mapping=True)

print(output)

print(tokenizer.convert_ids_to_tokens(output['input_ids']))

#['[CLS]', '〈', 'So', '##me', '##th', '##ing', '〉', '는', '조지', '해리', '##슨', '##이', '쓰', '##고', '비틀즈', '##가', '1969', '##년', '앨범', '《', 'Ab', '##be', '##y', 'Ro', '##ad', '》', '에', '담', '##은', '노래', '##다', '.', '[SEP]']

#{'input_ids': [2, 168, 30985, 14451, 7088, 4586, 169, 793, 8373, 14113, 2234, 2052, 1363, 2088, 29830, 2116, 14879, 2440, 6711, 170, 21406, 26713, 2076, 25145, 5749, 171, 1421, 818, 2073, 4388, 2062, 18, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

#'offset_mapping': [(0, 0), (0, 1), (1, 3), (3, 5), (5, 7), (7, 10), (10, 11), (11, 12), (13, 15), (16, 18), (18, 19), (19, 20), (21, 22), (22, 23), (24, 27), (27, 28), (29, 33), (33, 34), (35, 37), (38, 39), (39, 41), (41, 43), (43, 44), (45, 47), (47, 49), (49, 50), (50, 51), (52, 53), (53, 54),

(55, 57), (57, 58), (58, 59), (0, 0)]}

예를 들어, [CLS] 토큰은 원래 문장에 없던 토큰이기 때문에 beginindex와 endindex가 모두 0으로 반환됩니다. 그리고 ‘<’의 경우 원래 스트링의 index 0부터 index 1까지 있었기 때문에 (0, 1)이 반환됩니다.

Tokenizer로부터 반환된 offset mapping을 활용하여 Tokenized sequence내에서 원래의 entity가 포함된 토큰을 찾는 과정의 예시는 아래와 같습니다.

def find_entity(start_idx, end_idx, offsets):

   """

   KLUE relation extraction 데이터셋에 있는 entity로부터 변환된 tokens들의 위치를 반환합니다.

   """

   entity_token_indices = []

   found = False

   for idx, offset in enumerate(offsets):

       begin_offset, end_offset = offset

       if begin_offset <= start_idx < end_offset:

           found = True

       if found:

           entity_token_indices.append(idx)

       if begin_offset <= end_idx < end_offset:

           assert found

           found = False

       if len(entity_token_indices) != 0 and not found: break

   return entity_token_indices

subject = {"word": "비틀즈", "start_idx": 24, "end_idx": 26, "type": "ORG"}

results = find_entity(subject['start_idx'], subject['end_idx'], output['offset_mapping'])

entity_tokens = [e for i,e in enumerate(tokenizer.convert_ids_to_tokens(output['input_ids'])) if i in results]

print(entity_tokens)

#['비틀즈']

subject = {"word": "조지 해리슨", "start_idx": 13, "end_idx": 18, "type": "PER"}

results = find_entity(subject['start_idx'],subject['end_idx'], output['offset_mapping'])

entity_tokens = [e for i,e in enumerate(tokenizer.convert_ids_to_tokens(output['input_ids'])) if i in results]

print(entity_tokens)

#['조지', '해리', '##슨']

