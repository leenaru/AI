아래는 CLIP 모델 평가 시,  
1. **Confusion Matrix (혼동 행렬)**,  
2. **ROC Curve (수신자 조작 특성 곡선)**,  
3. **AUC (곡선 아래 면적)**  
을 계산하고 각각 시각화하는 코드 예시입니다.  

전체 코드는 JSON 파일로 구성된 데이터셋(각 샘플이  
```json
{ "image_path": "이미지경로", "label": "클래스명" }
```  
형식)을 불러와, CLIP 모델(예: "ViT‑B/32")과 지정한 텍스트 프롬프트를 사용하여 이미지의 임베딩을 계산합니다.  
예시에서는 **cosine similarity**를 유사도 측정 함수로 사용하여, 각 이미지에 대해 클래스별 유사도 점수를 얻고,  
이를 기반으로 예측 결과(각 샘플의 점수 벡터)를 활용해 혼동 행렬과 다중 클래스 ROC 곡선(클래스별 및 micro-average)을 계산합니다.

---

## 전체 코드

```python
import os
import json
import torch
import clip  # https://github.com/openai/CLIP
from PIL import Image
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ------------------------------------------------
# 1. 데이터셋 로드 함수
# ------------------------------------------------
def load_dataset(json_path):
    """
    JSON 파일에서 데이터셋을 불러옵니다.
    각 샘플은 {"image_path": "경로", "label": "클래스명"} 형식이어야 합니다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ------------------------------------------------
# 2. 유사도 측정 함수들
# ------------------------------------------------
def cosine_similarity(image_feature, text_features):
    """
    코사인 유사도 계산: 이미지 임베딩과 텍스트 임베딩 간의 cosine similarity를 계산합니다.
    점수가 높을수록 유사도가 높습니다.
    """
    return F.cosine_similarity(image_feature.unsqueeze(0), text_features, dim=1)

def dot_product_similarity(image_feature, text_features):
    """
    내적(도트 프로덕트)을 이용한 유사도 계산.
    """
    return torch.matmul(text_features, image_feature)

def euclidean_similarity(image_feature, text_features):
    """
    유클리드 거리를 기반으로 유사도를 계산합니다.
    거리가 짧을수록 유사도가 높으므로 음수의 거리를 score로 반환합니다.
    """
    distances = torch.norm(text_features - image_feature, dim=1)
    return -distances  # 음수 값을 반환

# ------------------------------------------------
# 3. 평가 함수: 예측 결과(예측 레이블과 연속적인 유사도 점수) 저장
# ------------------------------------------------
def evaluate_model_with_scores(model, preprocess, dataset, class_prompts, similarity_func, device):
    """
    model: CLIP 모델  
    preprocess: 이미지 전처리 함수  
    dataset: JSON 파일에서 로드한 데이터셋 (리스트)  
    class_prompts: 각 클래스에 대한 텍스트 프롬프트 리스트 (예: ["a photo of a cat", ...])  
    similarity_func: 유사도 측정 함수 (예: cosine_similarity)  
    device: "cuda" 또는 "cpu"
    
    각 샘플마다 예측된 label과, 각 클래스에 대한 유사도 점수(score vector)를 반환합니다.
    """
    with torch.no_grad():
        text_tokens = clip.tokenize(class_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    true_labels = []
    pred_labels = []
    score_list = []  # 각 샘플마다 (num_classes,) 크기의 유사도 score 벡터 저장
    
    for sample in dataset:
        image_path = sample["image_path"]
        true_label = sample["label"].strip().lower()
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"이미지 로드 에러 ({image_path}): {e}")
            continue
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        
        # 유사도 점수 계산 (예: cosine similarity)
        scores = similarity_func(image_feature.squeeze(0), text_features)
        scores_np = scores.cpu().numpy()   # shape: (num_classes,)
        pred_index = scores.argmax().item()
        # 프롬프트에서 "a photo of a " 제거하여 예측 클래스명 추출 (소문자 변환)
        pred_label = class_prompts[pred_index].replace("a photo of a ", "").strip().lower()
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        score_list.append(scores_np)
        
    # score_list를 (n_samples, num_classes) 형태의 numpy array로 변환
    score_array = np.vstack(score_list)
    return true_labels, pred_labels, score_array

# ------------------------------------------------
# 4. Confusion Matrix 시각화 함수
# ------------------------------------------------
def visualize_confusion_matrix(true_labels, pred_labels, unique_labels):
    """
    실제 레이블과 예측 레이블을 바탕으로 혼동 행렬을 계산하고, seaborn의 heatmap으로 시각화합니다.
    """
    # 각 레이블을 unique_labels에 따른 인덱스로 변환
    true_indices = [unique_labels.index(lbl) for lbl in true_labels]
    pred_indices = [unique_labels.index(lbl) for lbl in pred_labels]
    
    cm = confusion_matrix(true_indices, pred_indices, labels=range(len(unique_labels)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# ------------------------------------------------
# 5. ROC Curve 및 AUC 시각화 함수
# ------------------------------------------------
def visualize_roc_curve(true_labels, score_array, unique_labels):
    """
    다중 클래스 ROC 곡선 및 AUC를 계산하여 각 클래스별 ROC 곡선과 micro-average ROC 곡선을 시각화합니다.
    - true_labels: 실제 레이블 (리스트, 소문자)
    - score_array: (n_samples, num_classes) 크기의 유사도 점수 행렬
    - unique_labels: 고유 클래스 이름 (정렬된 리스트)
    
    ROC curve를 계산하기 위해, 먼저 true_labels를 one-hot 인코딩(이진화) 합니다.
    """
    # true_labels를 unique_labels의 인덱스로 변환
    true_indices = [unique_labels.index(lbl) for lbl in true_labels]
    # 이진화 (one-vs-all) 수행: 각 클래스에 대해 0/1 벡터 생성
    y_true_bin = label_binarize(true_indices, classes=range(len(unique_labels)))
    n_classes = len(unique_labels)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # 클래스별 ROC 곡선과 AUC 계산
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], score_array[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 클래스별 ROC 곡선 시각화
    plt.figure(figsize=(10,8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{unique_labels[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Each Class")
    plt.legend(loc="lower right")
    plt.show()
    
    # micro-average ROC curve 계산 및 시각화
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), score_array.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.figure(figsize=(8,6))
    plt.plot(fpr_micro, tpr_micro, label=f"Micro-average ROC (AUC = {roc_auc_micro:.2f})", color="navy")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# ------------------------------------------------
# 6. 메인 실행 함수
# ------------------------------------------------
def main():
    # JSON 파일 경로 (데이터셋)
    json_path = "dataset.json"  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 예시로 사용할 CLIP 모델 (여기서는 "ViT-B/32")와 전처리 함수 로드
    model_name = "ViT-B/32"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    
    # 데이터셋 로드
    dataset = load_dataset(json_path)
    # 데이터셋에서 사용되는 클래스 목록 추출 (소문자 정리)
    all_labels = [sample["label"].strip().lower() for sample in dataset]
    unique_labels = sorted(list(set(all_labels)))
    print("평가에 사용될 클래스:", unique_labels)
    
    # 텍스트 프롬프트 생성 (예: "a photo of a cat")
    class_prompts = [f"a photo of a {lbl}" for lbl in unique_labels]
    
    # 평가에 사용할 유사도 측정 함수 선택 (예: cosine similarity)
    similarity_func = cosine_similarity  # dot_product_similarity, euclidean_similarity 등도 가능
    
    # 평가: 각 이미지에 대해 예측 레이블과 각 클래스별 유사도 점수(score vector) 획득
    true_labels, pred_labels, score_array = evaluate_model_with_scores(model, preprocess, dataset, class_prompts, similarity_func, device)
    
    # ① Confusion Matrix 시각화
    visualize_confusion_matrix(true_labels, pred_labels, unique_labels)
    
    # ② ROC Curve 및 AUC 시각화
    visualize_roc_curve(true_labels, score_array, unique_labels)

if __name__ == "__main__":
    main()
```

---

## 코드 상세 설명

1. **데이터셋 로드 및 전처리**  
   - `load_dataset` 함수는 JSON 파일에서 이미지 경로와 레이블 정보를 읽어옵니다.

2. **유사도 측정 함수**  
   - `cosine_similarity`, `dot_product_similarity`, `euclidean_similarity` 함수는 이미지와 텍스트 임베딩 간의 유사도를 각각 계산합니다.  
   - 예시에서는 cosine similarity를 사용하지만, 다른 함수도 쉽게 교체할 수 있습니다.

3. **평가 함수 (`evaluate_model_with_scores`)**  
   - 이미지와 텍스트 임베딩을 계산한 후, 각 샘플에 대해 각 클래스별 유사도 점수 벡터를 저장합니다.  
   - 예측은 가장 높은 점수를 가진 클래스로 결정하며, 실제 레이블, 예측 레이블, 그리고 score vector를 반환합니다.

4. **Confusion Matrix 시각화**  
   - `visualize_confusion_matrix` 함수는 scikit-learn의 `confusion_matrix`를 사용하여 실제와 예측 인덱스를 비교한 후, seaborn의 heatmap으로 시각화합니다.  
   - 참고: [scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) citeturn0search0

5. **ROC Curve 및 AUC 시각화**  
   - `visualize_roc_curve` 함수는 다중 클래스 문제의 경우 각 클래스를 one-vs-all 방식으로 이진화하여 ROC 곡선과 AUC를 계산합니다.  
   - 각 클래스별 ROC 곡선과 함께 micro-average ROC 곡선도 함께 시각화합니다.  
   - 참고: [scikit-learn ROC Curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) citeturn0search0

6. **메인 실행 부분**  
   - JSON 데이터셋을 불러오고, CLIP 모델을 로드한 후 평가를 진행합니다.  
   - 최종적으로 혼동 행렬, ROC 곡선, 그리고 AUC를 시각화하여 평가 결과를 확인할 수 있습니다.

---

이 코드를 실행하기 전에 [CLIP GitHub 저장소](https://github.com/openai/CLIP), PyTorch, scikit-learn, matplotlib, seaborn 등이 설치되어 있어야 합니다.  
필요한 패키지가 없으면 `pip install clip torch scikit-learn matplotlib seaborn` 등의 명령어로 설치한 후 실행하시기 바랍니다.

혹시 추가적인 질문이나 코드 수정이 필요하면 언제든지 말씀해 주세요.
