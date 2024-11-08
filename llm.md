# LLM
**LLM 개발 과정**
- Data Preparation
  - Pre-Training Corpus
  - IF Corpus
  - Data Augmentation
  - Data Vectorization
    - Tokenizer
    - Word Embedding
- Modeling
  - Model Architecture
    - Transformer
  - Language Modeling
    - MLM, CLM
- Model Training
  - Pre-Training
    - Foundation Model
  - Fine-Tuning
    - Supervised Fine-Tuning
    - PEFT
    - RLHF
- LLM Inference
  - Text Generation
  - In-Context Learning
  - RAG
- Evaluation
  - Benchmark Dataset
  - Evaluation Metric

# NLP (Natural Language Processing)

> **Word $\subset$ Syntax $\subset$ Semantics $\subset$ Pragmatics**

* Corpus(말뭉치) : 자연어 처리에서 모델을 학습시키이 위한 데이터. 특정한 목적에서 표본을 추출한 집합
* Stop Words(불용어) : 분석과 관계없이 문장 내에서 많이 등장하는 단어. `"a"`, `"the"`, `"he"` 등
* Stemming(어간추출) : 단어를 기본 형태로 만드는 작업

## Corpus
언어 연구 및 자연어 처리에서 사용되는 대규모 텍스트 자료의 집합

### 특징
- **대규모 데이터** : 보통 수백만에서 수십억 단어 이상을 포함한 대규모 텍스트 데이터<br>
- **구조적 정리** : 대부분의 코퍼스는 분석이나 연구에 용이하도록 정리되어 있음. 예를 들어, 텍스트에 문장 구분, 단어 구분, 품사 태그 등을 붙여 사용자가 쉽게 분석할 수 있도록 구성됨.<br>
- **다양한 출처** : 특정 도메인(의학, 법률 등)이나 다양한 문체(소설, 뉴스, SNS 글 등)를 포함하여 언어의 여러 측면을 포괄할 수 있도록 구성

### 예제
- **Brown Corpus**: 1960년대에 만들어진 영어 코퍼스로, 다양한 장르의 글을 포함하고 있습니다. 현대 언어 연구의 초기 코퍼스로 유명합니다.
- **Wikipedia Corpus**: 위키백과의 내용을 담은 코퍼스로, 방대한 양의 정보와 다양한 주제를 포함하여 많은 NLP 모델의 학습 데이터로 사용됩니다.
- **SNS Corpus**: 트위터, 페이스북 등 소셜 미디어에서 수집된 데이터로, 최신 유행어와 자연스러운 대화체 연구에 적합합니다.

## Tokenizing
* 문장을 토큰으로 분할하는 과정
  * Word-based Tokenization : Vocabulary Size, Unknown Word 이슈
    * 시간이 | 화살처럼 | 지나간다
  * Character-based Tokenization : Semantic Meaning, Sequence Length 이슈
    * 시 | 간 | 이 | 화 | 살 | 처 | 럼 | 지 | 나 | 간 | 다 
  * Subword Tokenization : Rare words -> Subwords (Agglutinative Language)
    * 시간 | 이 | 화살 | 처럼 | 지나 | 간다

## BPE (Byte Pair Encoding)
* 단어의 빈도를 기반으로 가장 자주 등장하는 문자 쌍을 점차적으로 병합해 나가면서 텍스트를 토큰으로 나누는 방법
* 희귀 단어와 고유명사를 효과적으로 처리 (OOV)
* https://arxiv.org/pdf/1508.07909.pdf (2016)
* GPT, GPT-2, RoBERTa, BART, ...

### BPE example

> <details>
>  <summary>예시를 통한 BPE 동작 과정</summary>
>
> BPE는 가장 빈번하게 등장하는 연속된 문자 쌍을 반복적으로 병합하여 새로운 토큰을 만드는 알고리즘입니다[1][2].
> 
> ## 예시를 통한 BPE 동작 과정
> 
> 다음과 같은 문장 집합이 있다고 가정해보겠습니다:
> 
> ```
> ("hugs bun", 4)
> ("hugs pug", 1)
> ("hug pug pun", 4)
> ("hug pun", 6)
> ("pun", 2)
> ```
> 
> 괄호 안의 숫자는 각 문장의 빈도수입니다.
> 
> 1. **초기 단어 분리**: 
>    먼저 모든 단어를 개별 문자로 분리합니다.
>    
>    ```
>    ("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
>    ```
> 
> 2. **빈도수 높은 쌍 병합**:
>    가장 빈번한 문자 쌍을 찾아 병합합니다. 여기서는 "ug"가 20번으로 가장 많이 등장합니다.
> 
>    ```
>    ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
>    ```
> 
> 3. **반복**:
>    다음으로 빈도가 높은 쌍은 "un"으로 16번 등장합니다. 이를 병합합니다.
> 
>    ```
>    ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
>    ```
> 
> 4. **종료 조건**:
>    원하는 어휘 크기에 도달하거나 지정된 병합 횟수에 도달할 때까지 이 과정을 반복합니다[1].
> 
> ## 최종 어휘
> 
> 위 과정을 통해 생성된 최종 어휘는 다음과 같을 수 있습니다:
> 
> ```
> ("h", "ug", "p", "un", "b", "s")
> ```
> 
> 이렇게 생성된 어휘를 사용하여 새로운 텍스트를 토큰화할 수 있습니다[1].
> 
> ## BPE의 장점
> 
> 1. **유연성**: 알고리즘이 데이터에 따라 적응적으로 토큰을 생성합니다.
> 2. **OOV 문제 해결**: 미등록 단어(Out-of-Vocabulary) 문제를 효과적으로 해결할 수 있습니다.
> 3. **효율성**: 자주 사용되는 부분 단어를 효과적으로 인코딩합니다[2].
> 
> BPE는 이러한 방식으로 텍스트를 효율적으로 토큰화하며, 특히 기계 번역과 같은 자연어 처리 작업에서 널리 사용되고 있습니다.
>
> <details>
> <summary>Citations:</summary>
> [1] https://process-mining.tistory.com/189<br>
> [2] https://blog.naver.com/jjs1608/222882455728<br>
> [3] https://wikidocs.net/22592<br>
> [4] https://kaya-dev.tistory.com/46<br>
> [5] https://wikidocs.net/166825<br>
> [6] https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/<br>
> [7] https://velog.io/@yoonene/BPEByte-Pair-Encoding%EB%9E%80<br>
> [8] https://roytravel.tistory.com/162<br>
> </details>
</details>


## WordPiece
* Google 이 BERT 사전 학습을 위해 개발
* BPE와 동일하게 특수 토큰과 알파벳을 포함한 작은 Vocabulary 로 시작
* 접두사(##)를 추가하여 하위 단어 식별
* BPE의 빈도 대신 Likelihood 값이 가장 높은 쌍을 병합
* BERT, DistilBERT, MobileBERT

> <details>
> <summary>예시를 통한 WordPiece Tokenizer 동작 과정</summary>
>
> ### 동작 과정 예시
> 다음과 같은 간단한 코퍼스가 있다고 가정해봅시다:
> 
> ```Python
> "안녕하세요 자연어처리입니다"
> "자연어 처리는 재미있습니다"
> "처리 기술이 발전하고 있어요"
> ```
> 
> 1. 초기화
> 먼저 모든 단어를 개별 문자로 분리합니다:<br>
> `"안", "녕", "하", "세", "요", "자", "연", "어", "처", "리", "입", "니", "다", "는", "재", "미", "있", "습", "기", "술", "이", "발", "전", "하", "고", "있", "어", "요"`
> 
> 2. 빈도 계산
> 각 문자와 문자 쌍의 빈도를 계산합니다. 예를 들어:
> * "처리": 3번
> * "자연": 2번
> * "어": 2번
> 
> 3. 병합
> 가장 빈번한 문자 쌍을 하나의 새로운 토큰으로 병합합니다. 여기서는 "처리"가 가장 빈번하다고 가정하겠습니다.
> `"안", "녕", "하", "세", "요", "자", "연", "어", "처리", "입", "니", "다", "는", "재", "미", "있", "습", "기", "술", "이", "발", "전", "하", "고", "있", "어", "요"`
> 
> 1. 반복
> 원하는 어휘 크기에 도달할 때까지 2-3 단계를 반복합니다. 예를 들어, 다음 단계에서는 "자연"이 병합될 수 있습니다.
> `"안", "녕", "하", "세", "요", "자연", "어", "처리", "입", "니", "다", "는", "재", "미", "있", "습", "기", "술", "이", "발", "전", "하", "고", "있", "어", "요"`
> 
> ### 토큰화 예시
> 최종적으로 학습된 WordPiece 모델을 사용하여 새로운 문장 "자연어처리는 흥미롭습니다"를 토큰화해보겠습니다:<p>
> 1. "자연" (학습된 토큰)
> 1. "어" (학습된 토큰)
> 1. "처리" (학습된 토큰)
> 1. "는" (학습된 토큰)
> 1. "흥" (미등록 토큰, 개별 문자로 처리)
> 1. "미" (학습된 토큰)
> 1. "로" (미등록 토큰, 개별 문자로 처리)
> 1. "##습니다" (학습된 토큰, ##은 단어 중간이나 끝에 위치함을 나타냄)
> 
> 결과: `["자연", "어", "처리", "는", "흥", "미", "로", "##습니다"]`
> 
> 이렇게 WordPiece Tokenizer는 학습된 하위 단어를 사용하여 효율적으로 텍스트를 토큰화하며, 미등록 단어도 적절히 처리할 수 있습니다.
> </details>


## Unigram
* 빈도수가 아닌 Subword에 대한 확률 모델 사용
* Loss 는 Subword가 어휘 사전에서 제거되었을 경우 코퍼스의 Likelihood 값의 감소 정도
* ALBERT, T5, mBART, XLNet 


### 주요 토큰화 알고리즘 비교
| 알고리즘 | Traning | Training Step | Learns | Encoding |
|----------|-----------|------|------|---------|
| BPE | 작은 vocabulary 에서 시작<br>토큰을 결합하여 규칙 생성|the most common pair|Merge rules and a vocabulary|단어를 문자로 쪼개고 학습한 병합을 적용|
| WordPiece |  작은 vocabulary 에서 시작<br>토큰을 결합하여 규칙 생성 | the pair with the best score<br> based on the frequency of<br> the pair|Just a vocabulary|가장 긴 Subword 찾음|
| Unigram | 큰 vocabulary 에서 시작<br>토큰을 제거하여 규칙 생성|Loss를 최소화하도록<br> vocabulary에서 모든 토큰을 제거|토큰 별로 점수가 매겨진 vocabulary|학습한 점수를 사용하여<br> 가장 가능성이 높은 토큰 분할을 찾음


### 주요 토큰화 알고리즘의 장단점
| 알고리즘 | 주요 특징 | 장점 | 단점 | 사용 예 |
|----------|-----------|------|------|---------|
| BPE (Byte Pair Encoding) | 빈도수 기반으로 문자열 병합 | - OOV 문제 해결에 효과적<br>- 희귀 단어 처리 가능<br>- 언어에 독립적 | - 의미론적 정보 부족<br>- 과도한 분할 가능성 | GPT, RoBERTa |
| WordPiece | BPE와 유사하나 우도(likelihood) 기반 병합 | - OOV 문제 해결<br>- 단어의 의미 유지 | - 언어 의존적<br>- 긴 단어 처리 시 비효율적 | BERT, DistilBERT |
| Unigram | 언어 모델 기반 확률적 분할 | - 문맥을 고려한 토큰화<br>- 유연한 분할<br> - 어휘의 유연한 확장 가능 | - 계산 비용이 높음<br>- 구현이 복잡 | ALBERT, T5, mBART, XLNet
| 단어 기반 토큰화 | 공백이나 구두점으로 단어 분리 | - 구현 간단<br>- 직관적 | - OOV 문제<br>- 어휘 크기 증가<br> - 복합어, 접어 처리 미흡 | 초기 NLP 모델<br>간단한 NLP 처리 |
| 문자 기반 토큰화 | 개별 문자를 토큰으로 사용 | - OOV 문제 없음<br>- 작은 어휘 크기 | - 의미 정보 손실<br>- 시퀀스 길이 증가 | 일부 CNN 모델 |

## 자연어 처리 Library

### NLTK
[Python-based Natural Language Toolkit](https://www.nltk.org/) 


### KoNLPy
[Korean NLP in Python](https://konlpy.org/ko/latest/index.html) 

* 형태소 분석기 : Okt(Open Korea Text), Mecab, Komoran, Hannanum, Kkma
* 형태소 추출, 품사 태깅, 명사 추출, ...

```Python
!pip install konlpy

from konlpy.tag import Okt
okt = Okt()
print(okt.morphs("아버지가 방에 들어갑니다"))
```
```
['아버지', '가', '방', '에', '들어갑니다']
```

### transformers.AutoTokenizer
https://huggingface.co/docs/transformers/main_classes/tokenizer
#### Tokenizing

```python
from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.tokenize("Time files like an arrow!")
```
```
['time', 'files', 'like', 'an', 'arrow', '!']
```
#### Encoding, Decoding
```python
sentense = "Tokenizing a text is splitting it into words or subwords"
tokens = tokenizer.tokenize(sentense)
print(tokens)

print(tokenizer.vocab_size)
print(tokenizer.model_max_length)
print(tokenizer.model_input_names)

inputs = tokenizer(sentense)
encoded_sentence = inputs["input_ids"]
print(f"encoded_sentence\n{encoded_sentence}\n")

decoded_sentence = tokenizer.decode(encoded_sentence)
print(f"decoded_sentense\n{decoded_sentence}\n")
```
```
['token', '##izing', 'a', 'text', 'is', 'splitting', 'it', 'into', 'words', 'or', 'sub', '##words']
30522
512
['input_ids', 'attention_mask']
encoded_sentence
[101, 19204, 6026, 1037, 3793, 2003, 14541, 2009, 2046, 2616, 2030, 4942, 22104, 102]

decoded_sentense
[CLS] tokenizing a text is splitting it into words or subwords [SEP]
```

# Word Embedding

모든 것을 Vector로 표현

* Count Based : LSA(Latent Semantic Analysis), HAL(Hyperspace Analogue to Language)
* Prediction Based : CBOW, Skip-gram (Word2Vec, FastText)
* Co-occurrence Statistics (GloVe)

>$x_{embedding} = x_{one-hot} * W_{embeddig}$

## Traditional Vector Representation

### Sparse Representation: One-Hot Encoding
- Vocabulary Size가 벡터의 차원
- 벡터 중 한개 값만 1, 나머지는 0
- 단어의 의미를 표현하지 못함
- Curse of Dimensionality

### Dense Representation
- 밀집된 차원의 벡터로 표현, 벡터 값은 실수값
- 단어의 의미를 여러 차원에 분산하여 표현
- "비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다"
- 단어 벡터 간의 유의미한 유사도 계산 가능

### 비교
||On-Hot Vector|Embedding Vector|
|---|---|---|
|차원|고차원(사전 크기)|저차원(특징)|
|Sparae/Dense|Sparse|Dense|
|표현 방법|수동|훈련 데이터 학습|
|벡터값|0 or 1|실수|

### BoW(Bag-of-Words)
// TODO

## Word Embeddings

|특징|	Word2Vec|	FastText|	GloVe|
|---|---|---|---|
|모델 유형|	예측 기반(신경망 기반)|	예측 기반(신경망 기반) + 서브워드(n-gram)|	행렬 분해(Matrix Factorization)|
|학습 방식|	로컬 문맥 정보(CBOW, Skip-Gram)|	로컬 문맥 정보 + 단어를 n-gram 서브워드로 분해하여 학습	|단어 간 공기 빈도 행렬을 이용해 글로벌 통계 정보 반영|
|입력 단위|	단어|	단어 + 문자 단위 서브워드(n-gram)|	단어|
|OOV 처리|	OOV 처리 어려움|	서브워드를 활용하여 새로운 단어(OOV)도 처리 가능|	OOV 처리 어려움|
|형태 정보 활용|	형태 정보 반영 어려움|	형태소(서브워드) 단위로 분해하여 형태 정보 반영 가능	|형태 정보 반영 어려움|
|벡터 연산 가능성|	의미적 관계에 대한 벡터 연산 가능|	의미적 관계에 대한 벡터 연산 가능|의미적 관계에 대한 벡터 연산 가능|
|장점| + 자주 등장하는 단어 및 문맥을 효율적으로 학습<br>+ 계산이 상대적으로 빠름|+ 형태학적 정보 반영 가능<br>+ 희귀 단어와 OOV 처리에 강함|	+ 글로벌 통계 정보 반영<br>+ 대규모 말뭉치에서 일관성 높은 벡터 생성|
|단점|	- 형태 정보 및 OOV 처리 제한<br>- 희귀 단어 학습 제한적|	- 서브워드 처리를 위한 추가 연산이 필요<br>- 메모리 사용량이 많음|	- 공기 행렬 생성 및 처리 비용 큼<br>- 동적 학습 어려움|
|대표적인 사용 예|	Word2Vec 임베딩을 활용한 텍스트 분류, 감정 분석 등|	스펠링 오류 처리, 형태 복잡한 언어의 임베딩 생성|	대규모 사전 학습 임베딩, 텍스트 분류 및 유사도 측정|


### Word2Vec
https://code.google.com/archive/p/word2vec/
1. 개요
     - Google에서 개발한 단어 임베딩 기법
     - 단어의 의미를 고정된 길이의 실수 벡터로 표현
     - 비슷한 의미의 단어들은 벡터 공간에서 가까이 위치하도록 학습
2. 학습 방식
     - a) CBOW (Continuous Bag of Words):
       - 주변 단어들을 입력으로 중심 단어를 예측
       - 예: "The (?) cat sat on the mat" - 주변 단어로 '?'에 올 단어 예측
     - b) Skip-Gram:
       - 중심 단어를 입력으로 주변 단어들을 예측
       - 예: "cat"이라는 단어로 주변에 올 수 있는 단어들 예측
3. 네트워크 구조
    - 얕은 신경망 구조 (1개의 은닉층)
    - Input Layer : One-hot encoded vector
    - Hidden Layer : 룩업 테이블 연산, 활성화 함수 없음
    - Output Layer : Softmax 를 통한 단어 확률 계산
4. 학습 과정
    - 윈도우 크기를 정해 문장을 슬라이딩하며 학습 데이터 생성
    - 손실 함수 최소화를 통해 가중치 행렬 W와 W' 학습
    - 학습된 W (또는 W와 W'의 평균)를 최종 단어 임베딩으로 사용
5. 장점
    - 대규모 텍스트 데이터에서 효율적으로 학습 가능
    - 단어 간의 의미적, 문법적 관계를 잘 포착
    - 벡터 연산을 통한 단어 관계 추론 가능 (예: king - man + woman ≈ queen)
6. 한계
    - 동음이의어 처리의 어려움
    - 문맥을 고려하지 않은 고정된 벡터 표현

| 특징 | CBOW (Continuous Bag of Words)	| Skip-Gram |
|---|---|---|
|학습 방식| 문맥 단어를 입력으로 하여 중심 단어를 예측|	중심 단어를 입력으로 하여 주변 문맥 단어를 예측|
|입력과 타겟| 문맥 단어들을 입력으로 사용, 중심 단어를 타겟으로 예측|	중심 단어를 입력으로 사용, 문맥 단어들을 타겟으로 예측|
|학습 속도|	문맥 단어의 평균 벡터를 계산하여 학습 속도가 빠름|	다수의 문맥 단어를 예측해야 하므로 학습 속도가 느림
|적합한 데이터|	자주 등장하는 단어 예측에 효과적|	희귀 단어 예측에 효과적|
|장점|+ 학습이 빠름<br>+ 데이터 요구량이 적음|+ 희귀 단어 및 다양한 문맥 정보 학습 가능|
|단점|- 문맥 단어의 순서 정보 손실 가능|	- 계산량이 많아 학습 속도가 느림|
|적용 예시|	간단한 텍스트 분석, 자주 등장하는 단어 임베딩 학습|	복잡한 문장 구조, 희귀 단어의 의미 학습|
|사용 예|	BERT의 사전 학습, 간단한 텍스트 기반 임베딩|	Word2Vec, FastText, 다양한 NLP 과제에서의 임베딩 학습


### FastText
https://github.com/facebookresearch/fastText

#### 특징
- Facebook AI Research 팀이 개발한 단어 임베딩 모델
- Word2Vec의 한계를 극복하기 위해 개발
- 단어를 벡터화할 때 서브워드(subword) 정보를 활용하여 단어를 단순히 하나의 단위로 취급하지 않고, <b>n-gram(문자 단위 조각)</b>을 포함하여 보다 세밀한 단어 표현을 생성
- 희귀 단어나 새로운 단어도 잘 학습하고, 단어의 형태와 어근, 접사 정보를 반영할 수 있음
- 형태가 복잡한 언어(예: 한국어, 일본어, 핀란드어)나, 스펠링 실수가 잦은 데이터에서 강력한 성능을 발휘
  - 한국어에서 "자연어처리"와 "자연어"는 철자 유사도가 높아 벡터 공간에서도 가깝게 위치
  - 형태소 기반 정보를 통해, 문법적 변형이나 새로운 단어에도 유연하게 대처할 수 있어 언어 모델에서 널리 활용


#### 장점
- 철자나 어근, 접두사, 접미사 등의 문자 패턴을 학습하여 형태학적 정보가 풍부한 언어(예: 한국어)에서도 강력한 성능을 발휘
- 새로운 단어가 등장하더라도 n-gram을 이용해 효과적인 벡터를 생성할 수 있어, 희귀 단어(OOV)에 강함.
- 단어의 어휘적 의미뿐 아니라, 철자 기반의 유사성까지 반영할 수 있습니다.

#### 단점
- 서브워드 정보까지 포함하기 때문에, Word2Vec에 비해 학습 속도가 느리고 메모리 사용량이 더 많을 수 있음.
- 매우 긴 단어는 많은 서브워드를 생성하게 되어 오히려 학습이 비효율적일 수 있음.

<br><br>

> <details>
> <summary><i>예시를 통한 FastText 동작 과정 (1)</i></summary>
> 
> #### 1. 서브워드(subword) 정보 활용:<br>
> FastText는 단어를 더 작은 단위인 서브워드로 나누어 처리 예를 들어:<br>
> "apple"이라는 단어가 있을 때 (n=3~6으로 설정 시):<br>
> 3-gram: <code>ap, app, ppl, ple, le</code><br>
> 4-gram: <code>app, appl, pple</code><br>
> 5-gram: <code>appl, apple</code><br>
> 6-gram: <code>apple</code><br>
> 전체 단어: <code>apple</code><br>
> 이렇게 나눈 서브워드들을 모두 벡터화하여 학습<br>
> 
> #### 2. OOV(Out-of-Vocabulary) 문제 해결:<br>
> 학습 데이터에 없던 단어도 서브워드 정보를 이용해 벡터를 생성할 수 있음 예를 들어:<br>
> "birthplace"라는 단어를 학습하지 않았더라도, <br>"birth"와 "place"의 서브워드 정보를 이용해 "birthplace"의 벡터를 추정할 수 있음
> </details>

<br>

> <details>
> <summary><i>예시를 통한 FastText 동작 과정 (2)</i></summary>
> <br>
> 
> ```Python
> "자연어 처리는 매우 흥미롭습니다."
> ```
>
> #### 1. 단어를 서브워드로 분해<br>
> FastText는 각 단어를 문자 단위의 n-gram으로 쪼개어 학습. 예를 들어 n-gram 크기를 3으로 설정했다면, "자연어"는 다음과 같은 서브워드들로 분해:
> 
> * 자연어의 3-gram:
>   * 자연어 자체
>   * 자연
>   * 연어
> 
> 이와 같은 방식으로, "자연어"의 n-gram은 `자`, `자연`, `연어`, `어`와 같이 여러 문자 조합으로 표현
>
> #### 2. 각 서브워드에 임베딩 벡터 생성
> FastText는 각 서브워드의 벡터를 개별적으로 학습합니다. 예를 들어, "자연어"라는 단어는 `자`, `자연`, `연어`, `어`라는 n-gram 벡터의 조합으로 표현됩니다. 이는 단어가 여러 서브워드의 벡터로 이루어진 합산 벡터로 표현된다는 의미입니다.
> 
> #### 3. 문맥에서의 학습
> FastText는 Word2Vec의 CBOW 또는 Skip-Gram 모델을 바탕으로 학습됩니다. 예를 들어, "처리"라는 단어가 "자연어"와 "매우" 사이에 있을 때, "처리"가 중심 단어가 되어 주변 문맥 단어를 예측하거나 반대로 주변 단어로 중심 단어를 예측하는 방식으로 학습됩니다. 이때 "처리"라는 단어 역시 `처`, `처리`, `리` 등의 n-gram으로 쪼개져서 학습에 활용됩니다.
> 
> 이 과정에서 FastText는 단어 자체뿐만 아니라 단어 내 문자 패턴에 대한 정보를 학습하게 되어, 새로운 단어나 철자가 비슷한 단어에 대해서도 효과적인 벡터 표현을 생성할 수 있게 됩니다.
> 
> #### 4. 새로운 단어 처리
> FastText의 강력한 점은 처음 보는 단어에도 유용한 임베딩을 생성할 수 있다는 것입니다. 예를 들어 "자연어처리"라는 단어가 새로 등장하면, 이 단어를 `자연어`, `연어처`, `어처리`와 같은 n-gram으로 쪼개고, 각 n-gram 벡터의 합산을 통해 "자연어처리"의 벡터를 생성할 수 있습니다. 이는 기존의 Word2Vec이나 GloVe와 같은 모델에서 새로운 단어를 OOV(out-of-vocabulary)로 처리하는 한계를 극복하는 중요한 장점입니다.
>
> </details>

### GloVe
https://github.com/stanfordnlp/GloVe

#### 특징
- 사전 학습된 벡터를 다양한 NLP 과제에 활용
  - 예를 들어 감정 분석, 텍스트 분류, 기계 번역, 질의 응답 시스템 등에서 단어 임베딩으로 사용되며, 사전 학습된 GloVe 벡터를 불러와 Fine-Tuning하여 모델의 성능을 향상시킬 수 있음
- 전체 텍스트의 통계적 정보를 반영한 단어 벡터를 생성하며, 이를 통해 단어 간의 의미적 관계를 수치적으로 표현할 수 있음.
- Word2Vec과는 달리 글로벌한 말뭉치 통계를 반영하므로 더 일관성 있는 벡터를 학습할 수 있으며, 단어 간 관계를 벡터 연산으로 나타내는 데 강점

#### 장점
- 전반적 통계 정보 반영: 전체 말뭉치의 공기 빈도 정보를 반영하므로, 단어 벡터가 전체 텍스트에 대한 일관성을 가집니다.
- 단어 간 의미적 관계를 벡터 연산으로 표현: 벡터 간의 덧셈과 뺄셈을 통해 단어 간의 관계를 수치적으로 나타낼 수 있습니다.
- 대규모 사전 학습 데이터 사용에 적합: 대규모 말뭉치에서 학습한 단어 벡터는 다양한 NLP 작업에서 높은 성능을 보입니다.

#### 단점
- Co-occurrence 행렬의 크기 문제: 대용량 데이터에서 단어 공기 행렬을 생성하고 최적화하는 데 상당한 메모리와 계산 비용이 필요합니다.
- 동적 학습 어려움: Word2Vec과 달리 GloVe는 새로운 단어를 동적으로 학습하기 어렵습니다.


## Contextual Embeddings

Transformer-based Models

- ELMo
- BERT
- GPT
