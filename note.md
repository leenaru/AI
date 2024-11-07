# AI
Generative AI $\subset$ Deep Learning $\subset$ Machine Learning $\subset$ AI

- Machine Learning : 데이터에서 규칙을 찾기
- Deep Learning : 데이터를 계층적으로 학습하여 패턴 찾기

# Procedure
- 문제 정의
- 데이터 수집
- 데이터 전처리
- 모델 선택
- 모델 학습
- 모델 평가
- 모델 개선
- 예측 및 적용



# DNN (심층신경망 (DNN)

### SLP (Single Layer Perceptron, 단층퍼셉트론)

```
nn.Linear
```
Fully Connected Layer

### MLP (Multi Layer Perceptron, 다층퍼셉트론)


### Activation Function (활성화 함수)  
- 활성화함수는 인공신경망(Artificial Neural Network)에서 매우 중요한 역할을 하는 요소로,
- 뉴런(또는 노드)들이 입력받은 값을 비선형으로 변환하여 다음 층으로 전달해주는 역할. 
- 이러한 비선형 변환을 통해 신경망은 복잡한 데이터 패턴을 학습할 수 있음.


|활성화함수|정의|특징|장점|단점|
|---|---|---|---|---|
|**Sigmoid**|$\sigma(x) = \frac{1}{1 + e^{-x}}$|<li>출력값이 항상 0과 1 사이에 위치하여, 확률을 나타내는 문제에 자주 사용<br>출력값이 작을수록 0에 가까워지고, 클수록 1에 가까워짐<li>주로 이진 분류 문제에서 마지막 출력 층에 사용|<li>비선형성을 가지므로 복잡한 패턴을 학습할 수 있음<li>출력을 확률 값으로 해석할 수 있어 분류 문제에서 유용|<li>Vanishing Gradient 문제: 입력 값이 매우 크거나 매우 작으면 기울기(미분값)가 거의 0이 되어서 학습이 잘되지 않음<li>출력이 항상 양수이기 때문에, 중간 층에서 기울기 소실 발생 가능
|**ReLU**(Rectified Linear Unit)|$f(x)=max(0,x)$|입력 값이 0보다 작으면 0, 0보다 크면 입력값 그대로를 반환. 비선형성|<li>계산이 빠름: 다른 함수대비 계산이 간단하여 학습 속도가 빠름<li>Gradient 소실 문제 감소: 시그모이드와 달리 기울기 소실 문제를 어느 정도 해결할 수 있음<li>Sparse Activation: 음수 값에서는 출력을 0으로 만들어, 일부 뉴런이 활성화되지 않는 특성을 가짐. 이는 모델이 불필요한 뉴런을 활성화시키지 않아 효율적.|Dying ReLU 문제: 입력값이 음수일 경우, 기울기가 0이 되어 뉴런이 학습되지 않는 문제가 발생할 수 있음. 이로 인해 일부 뉴런이 비활성화될 가능성


# CNN (Convolution Neural Network, 합성곱 신경망)
주로 이미지 인식, 영상 분석, 객체 탐지 등 시각적 데이터를 처리
