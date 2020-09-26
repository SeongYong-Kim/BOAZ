# Efficient Estimation of Word Representations in Vector Space



## 1. Introduction

성능 면에서 엄청난 양의 데이터로 훈련된 간단한 모델들이 적은 데이터로 훈련된 복잡한 모델을 능가한다. N-gram 모델이 대표적인 간단한 모델의 예이다. 하지만 이러한 모델들은  자동 음성 인식과 같은 많은 데이터를 필요로 하는 영역에서는 한계를 갖는다. 최근의 머신러닝 기술의 발전과 함께 더욱 복잡한 모델을 매우 대량의 데이터 셋으로 학습 가능해졌고 그 중 성공적인 방법은 단어의 분포를 표현한 것이다. 예를 들어 언어 모델을 기반으로 하는 신경망 구조는 N-gram을 상당히 능가한다.



### 1.1 Goals of the Paper

수십억의 단어와 어휘집에 존재하는 수백만의 단어들로 이루어진 데이터로부터 높은 수준의 word vector를 학습하는 기술을 소개하는 것이다. 이 학습방법은 오직 비슷한 단어가 서로 근접하는 것 뿐만 아니라 단어가 복수의 유사도를 갖도록 한다. 이를 통해 단어 간격 기술을 이용해 ‘King’ - ‘Man’ + ‘Woman’ = ‘Queen’과 같은 단어 벡터의 대수적 연산을 실행할 수 있다. 결론적으로 단어의 선형 규칙들을 보존하는 새로운 모델 구조를 개발하여 word vector 표현의 정확도를 높이려한다.



### 1.2 Previous Work

NNLM

Linear Projection Layer와 Non-Linear Hidden Layer를 기반으로 word vector 표현과 통계학적인 언어 모델의 결합을 학습하는데 사용된다. 계산 복잡도가 크다.



## 2. Model Architectures

계산 복잡도(O) = E * T * Q

E : Epoch 수(보통 3~50)

T : training set의 단어 수(보통 10억)

Q : 모델 구조에 의해 결정



### 2.1 NNLM

V차원, N개의 이전 단어

계산 복잡도(Q) = N * D + N * D * H + H * V

H * V의 계산이 큰데 hierarchical softmax??? 또는 정규화 되지 않은 모델을 완전히 정규화하는 것을 피하여 줄일 수 있다. 따라서 가장 큰 복잡도는 N * D * H에 의해서 발생한다.

Word2Vec은 어휘가 Huffman 이진 트리로 표현되고 hierarchical softmax를 사용한다. 이는 neural network가 단어의 특징을 추출하는데 효과적이라는 관찰 결과를 따른 것이다.  Huffman 트리는 단어의 빈도에 짧은 이진 코드를 할당하고 이는 검증되어야 하는 output 수를 줄인다. 이 같은 방법이 NNLM에서 문제되는  N * D * H 부분의 계산 복잡도를 줄이지는 않지만 hidden layer를 사용하지 않는 구조로 설계된다.



### 2.2 RNNLM

Q = H * H + H * V

H * V항은 hierarchical softmax를 이용해 H * log2(V)로 줄어든다. 따라서 가장 큰 복잡도는 H * H에 의해 발생한다.



### 2.3 Parallel Training of Neural Networks

큰 데이터 셋으로 모델을 훈련시키기 위해 distributed framework인 DistBelief를 사용했다.

이 framework는 병렬로 같은 모델에 대한 여러개의 복제본을 실행할 수 있게 한다. 복제본은 모든 변수를 보유하는 중앙 집중식 서버에서 gradient 갱신을 동기화한다. 병렬 훈련을 위해서 Adagrad라고 불리는 Adaptive learning rate 방식을 기반으로 하는 mini-batch 비동기식 gradient descent를 사용한다.



## 3. New Log-linear Models

이전 섹션에서 모델의 비선형 hidden layer가 가장 큰 계산 복잡도의 원인이라는 것을 파악했다. 이 섹션에서 계산 복잡도를 최소화 하기 위해 두가지 새로운 모델 구조를 제안한다.



### 3.1 Continuous Bag-of-Words Model(CBOW)

비선형 hidden layer가 사라지고 projection 층이 모든 단어들에서 공유되는 것을 제외하면 NNLM과 유사하다. 또 다른 점은 NNLM은 예측하고자 하는 단어 앞자리의 단어들만 예측에 사용했다면 CBOW는 앞으로 나올 단어들도 사용한다.  이때 계산복잡도는

Q = N * D + D * log2(V)

와 같이 표현된다. 모델 구조는 아래와 같다.



### 3.2 Continous Skip-gram Model

Skip-gram은 CBOW와 비슷하지만 문맥을 통해서 현재 단어를 예측하는 것 대신에 한 단어를 통해 그 주변 단어들을 예측한다. 각각의 단어를 연속적인 projection layer와 함께 log-linear 분류기의 input으로 이용한다. 또한 멀리 떨어진 단어일수록 현재 단어와 적은 연관을 갖기 때문에 적은 샘플링을 하는 것으로 적은 가중치를 두었다.

Skip-gram의 계산 복잡도(Q) = C × (D + D × log2(V)), C : 단어의 최대 거리

만약 C = 5이라면 각각의 훈련 단어를 위해서 우리는 1:C 범위에서 랜덤하게 숫자 R을 선택한다. 그리고 앞 뒤로 R개의 단어를 정답 라벨링한다.ouput은 R + R 개의 단어이다.

우리의 실험에서는 C = 10 을 사용한다.

아래는 CBOW와 Skip-gram의 모델구조이다.

![Image for post](https://miro.medium.com/max/1986/1*7NwyAZllQL9S9_dj7jfVcA.png)



### 4. Results

Word vector에서 산술 연산을 실행할 수 있다. 예를들면 smallest를 찾기 위한 산술과정으로 X = vector(“biggest”) — vector(‘big”) + vector(“small”) 를 계산할 수 있다. 또한 매우 많은 데이터를 가지고 높은 차원의 단어 벡터를 학습할 때 결과 벡터는 매우 미묘한 의미의 관계를 답변하는데 사용될 수 있다는 것을 알았다. 예를 들어 프랑스 - 파리, 독일 - 베를린과 같은 국가와 도시의 관계가 있다

![Image for post](https://miro.medium.com/max/1731/1*erHM5-TM7ewHB9V18DUogg.png)



### 4.1 Task Description

Word vector의 quality를 측정하기 위해 5가지 종류의 의미론적인 테스트 셋, 9가지 종류의 구문론적인 질문들을 담는 테스트 셋을 정의했다. 전체적으로 보아 8869개의 의미론적인 질문들과 10675개의 구문론적인 질문들이 있다. 각각의 카테고리에서 이 질문들은 두 가지 단계에 의해 생성된다.

첫 번째, 직접 유사한 단어들의 쌍의 목록을 작성한다.

두 번째, 두 단어의 쌍을 랜덤으로 뽑아 2.5K 개의 문제를 제작했다.

위의 방법으로 얻어낸 word vector와 가장 가까운 word vector가 정확하게 질문의 답변과 같을 때를 정확하게 답변된 것이라고 본다. 동의어 등은 단어 형태학에 대한 정보가 입력되지 않은 현재의 모델로는 100%의 정확도가 불가능할 것이라는 것을 의미하지만 긍정적인 상관관계를 갖을 것으로 생각된다.



### 4.2 Maximization of Accuracy

구글 뉴스 코퍼스를 word vector training에 사용하였다. 이 코퍼스는 60억 단어들을 포함하고 그 중 가장 많이 사용된는 단어 100만개를 추출했다. Training과정에서 많은 데이터, 높은 차원의 word vector를 사용하는 것이 높은 정확도를 나타낼 것으로 예상되는 constrained optimization 문제를 직면하였다. 가능한 빠르게 결과를 얻기 위해서 가장 많이 사용된 3만 개의 단어로 word를 제한하여 모델을 평가했다.

![Image for post](https://miro.medium.com/max/1781/1*z0mYhDrNVSgWD_c9AYrhbQ.png)

이 표는 개별적으로 데이터 또는 차원을 증가시켰을때, 성능향상이 절감됨을 볼 수 있고 이는 동시에 벡터의 차원과 데이터의 양을 증가시켜야 더 좋은 결과를 낸다는 것을 알 수 있다. 위의 표는 Stochastic Gradien Descent와 Backpropagation을 3 cycle  진행한 결과이고 초기 학습률은 0.025, 이는 선형적으로 감소하다가 마지막 cycle에서 0에 근접하게 된다.



### 4.3 Comparison of Model Architectures

같은 training set으로 640차원의 word vector로 학습한 결과를 비교한다. 모델 별 의미론적, 구조론적 accuracy는 아래의 표와 같다.

![Image for post](https://miro.medium.com/max/2030/1*bb37mNsYRXzp1-ObUukHGg.png)

Skip-gram은 의미론적으로 다른 모델들에 비해 월등히 좋다.

다음으로 모델들을 한개의 CPU만으로 훈련하여 평가한다.

![Image for post](https://miro.medium.com/max/1868/1*JTIazknKC3oFbddfPfwuHA.png)

Skip-gram 모델이 3일치, CBOW 모델은 하루 구글 뉴스 데이터의 일부로 훈련한다.

같은 데이터로 3번의 Cycle을 훈련하는 것보다 두 배 이상의 데이터로 1번 Cycle을 훈련시키는 것이 더 좋은 데이터를 갖는다. 그리고 더 빠르다.



### 4.4 Large Scale Parallel Training of Models

이전에 언급했든 DistBelif라고 불리는 distributed framework에서 다양한 모델을 실행한다. Training 시에 50~100 개의 복제 모델을 사용한다. Distributed framework 위에 있기 때문에 CBOW 모델과 Skip-gram 모델의 CPU 사용은 한개의 머신에서 수행하는 것보다 더욱 서로에게 가까워진다.



### 4.5 Microsoft Research Sentence Completion Challenge

이 작업은 각각 문장에 한개의 글자가 빠져있는 1040개의 문장을 사용한다. 목표는 5개의 합리적인 선택지 중에 문장에 가장 어울리는 단어를 선택하는 것이다. 이 작업에서 Skip-gram 구조의 성능을 확인했다.

첫번째로 우리는 [MRSCC]에 있는 5천만개의 단어를 640 차원 모델로 training했다. 그리고 우리는 임의 단어의 주변 단어를 예측하여 각각 모르는 단어를 문장에 대한 답을 계산했다. 마지막 문장 답은 각각의 예측의 합으로 결정한다.

![Image for post](https://miro.medium.com/max/1066/1*BhO5o8o7ZKSGhWuf1X5uhg.png)

Skip-gram 모델 자체로는 LSA 유사도 보다 좋은 성능을 내지 못하지만 RNNLM으로 얻은 값으로 이 모델을 보완하고 가중치 조합은 58.9%의 정확도를 나타낸다.



### 5. Examples of the Learned Relationships

아래 표는 단어의 다양한 관계를 보여준다.

![Image for post](https://miro.medium.com/max/1961/1*6cqmIpG_UXmigDw60cB0Jg.png)

두 단어 벡터의 뺄셈으로 관계가 정의된다. 그리고 결과는 다른 단어와 더해질 수 있다. 예를 들어 Paris — France + Italy = Rome 의 연산이 가능하다. 우리는 더 많은 차원, 데이터를 가지고 학습시에 훨씬 좋은 성능을 낼 것으로 예상한다. 정확도를 높이는 또 다른 방법은 한개 이상의 관계를 공급하는 것이다. 관계를 10개 주므로써 10% 정도의 정확도 향상을 관찰했다. 



### 6. Conclusion

* 인기있는 모델인 신경망 모델과 비교했을 때 매우 간단한 모델 구조를 사용하여 얻은 단어 벡터가 좋은 성능을 가질 수 있음을 확인했다.
* 거대한 데이터 셋이 있다면 매우 적은 계산 복잡도로 높은 차원의 단어 벡터를 정확하게 구할 수 있다.
* DistBelief distributed framework를 사용하여 일반적으로 크기의 한계가 없는 어휘들을 1조개의 단어가 들어있는 말뭉치로 CBOW와 Skip-gram 모델을 학습할 수 있었다.
* 이 논문에서 제시한 모델을 통해 word vector들을 기반으로 하는 신경망 구조들은 감정 분석이나 비유 탐색 등의 NLP 작업에 도움을 줄 것으로예상한다.

