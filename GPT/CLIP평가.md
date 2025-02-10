아래는 이전에 작성한 코드에 평가 지표(Accuracy, Precision, Recall, F1)를 계산하여 출력하고, 모델별/유사도 측정 방식별 결과를 시각화하는 전체 파이썬 코드 예시입니다.  
전체 코드는 JSON 파일로 구성된 데이터셋을 읽어 CLIP 모델 3종(예: ViT‑B/32, RN50, ViT‑B/16)과 세 가지 유사도 측정(코사인, 내적, 유클리드 기반)을 비교 평가한 후, 각 평가 지표 값을 출력하고 matplotlib과 seaborn을 이용해 grouped bar chart 형태로 시각화합니다.

---

## 요약

- **데이터셋:** JSON 파일  
  각 항목은  
  ```json
  {
      "image_path": "경로/이미지.jpg",
      "label": "클래스명"
  }
  ```  
  의 형식이어야 합니다.
- **모델:** 최소 3종 CLIP 모델  
- **유사도 측정:**  
  - 코사인 유사도  
  - 내적 (Dot Product)  
  - 유클리드 거리 기반 (음수값으로 score 변환)
- **평가지표:** Accuracy, Precision, Recall, F1 (다중 클래스의 경우 average='macro' 사용)
- **시각화:** 모델 및 유사도 측정 방식별 평가 지표를 grouped bar chart로 표현

---

## 전체 코드

```python
import os
import json
import torch
import clip  # https://github.com/openai/CLIP
from PIL import Image
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 1. 데이터셋 로드 함수
# ---------------------------
def load_dataset(json_path):
    """
    JSON 파일에서 데이터셋을 불러옵니다.
    각 샘플은 {"image_path": "경로", "label": "클래스명"} 형식이어야 합니다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ---------------------------
# 2. 유사도 측정 함수들
# ---------------------------
def cosine_similarity(image_feature, text_features):
    """
    코사인 유사도 계산 (높을수록 유사도가 높음)
    """
    # image_feature: [D], text_features: [num_classes, D]
    return F.cosine_similarity(image_feature.unsqueeze(0), text_features, dim=1)

def dot_product_similarity(image_feature, text_features):
    """
    단순 내적을 통한 유사도 계산 (내적 값이 높을수록 유사도가 높음)
    """
    return torch.matmul(text_features, image_feature)

def euclidean_similarity(image_feature, text_features):
    """
    유클리드 거리 기반 유사도 계산:
    거리가 짧을수록 유사도가 높으므로, 음수의 거리를 유사도 score로 사용
    """
    distances = torch.norm(text_features - image_feature, dim=1)
    return -distances  # 음수 값을 반환하여 거리가 짧을수록 높은 score가 됨

# ---------------------------
# 3. 모델 평가 함수 (평가지표 출력을 위한 예측값 저장)
# ---------------------------
def evaluate_model(model, preprocess, dataset, class_prompts, similarity_funcs, device):
    """
    model: CLIP 모델
    preprocess: 이미지 전처리 함수
    dataset: JSON에서 로드한 데이터셋 (리스트)
    class_prompts: 각 클래스에 대한 텍스트 프롬프트 리스트 (예: ["a photo of a cat", ...])
    similarity_funcs: {"유사도_이름": function, ...}
    device: "cuda" 또는 "cpu"
    
    각 유사도 함수에 대해 예측값과 실제값을 리스트에 저장하여 반환
    """
    # 텍스트 임베딩 계산 (한번만 수행)
    with torch.no_grad():
        text_tokens = clip.tokenize(class_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 결과 저장: 각 유사도 방법별로 {'pred': [예측값], 'true': [실제값]} 저장
    results = {name: {"pred": [], "true": []} for name in similarity_funcs.keys()}
    
    for sample in dataset:
        image_path = sample["image_path"]
        true_label = sample["label"].strip().lower()
        
        # 이미지 로드 및 전처리
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"이미지 로드 에러 ({image_path}): {e}")
            continue
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # 이미지 임베딩 계산
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        
        # 각 유사도 함수에 대해 예측 수행
        for sim_name, sim_func in similarity_funcs.items():
            sims = sim_func(image_feature.squeeze(0), text_features)
            pred_index = sims.argmax().item()
            # 간단한 전처리: 프롬프트에서 "a photo of a " 제거하여 클래스명 추출
            pred_label = class_prompts[pred_index].replace("a photo of a ", "").strip().lower()
            
            results[sim_name]["pred"].append(pred_label)
            results[sim_name]["true"].append(true_label)
            
    return results

# ---------------------------
# 4. 평가지표 계산 함수
# ---------------------------
def compute_metrics(true_labels, pred_labels, labels):
    """
    true_labels: 실제 레이블 리스트
    pred_labels: 예측 레이블 리스트
    labels: 전체 클래스 리스트 (정렬된 상태)
    
    Accuracy, Precision, Recall, F1 (macro average) 계산
    """
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, labels=labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, labels=labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, labels=labels, average='macro', zero_division=0)
    # classification_report_str = classification_report(true_labels, pred_labels, labels=labels)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ---------------------------
# 5. 시각화 함수
# ---------------------------
def visualize_metrics(overall_metrics):
    """
    overall_metrics: {model_name: {sim_method: {accuracy, precision, recall, f1}}}
    각 평가 지표별로 모델 및 유사도 방식에 따른 값을 grouped bar chart로 시각화
    """
    data = []
    for model_name, sim_dict in overall_metrics.items():
        for sim_name, metrics in sim_dict.items():
            data.append({
                "Model": model_name,
                "Similarity": sim_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"]
            })
    df = pd.DataFrame(data)
    
    # 시각화: Accuracy, Precision, Recall, F1 별로 subplot 생성
    metrics_list = ["Accuracy", "Precision", "Recall", "F1"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        sns.barplot(x="Model", y=metric, hue="Similarity", data=df, ax=ax)
        ax.set_title(metric)
        ax.set_ylim(0, 1)  # 평가 지표가 0~1 사이의 값이므로
    plt.tight_layout()
    plt.show()

# ---------------------------
# 6. 메인 실행 부분
# ---------------------------
def main():
    # 설정
    json_path = "dataset.json"  # JSON 데이터셋 파일 경로
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 비교할 CLIP 모델 목록 (예: ViT-B/32, RN50, ViT-B/16)
    clip_models = ["ViT-B/32", "RN50", "ViT-B/16"]
    
    # 유사도 함수들을 딕셔너리로 저장
    similarity_funcs = {
        "Cosine": cosine_similarity,
        "DotProduct": dot_product_similarity,
        "Euclidean": euclidean_similarity
    }
    
    # 데이터셋 로드
    dataset = load_dataset(json_path)
    if not dataset:
        print("데이터셋을 불러올 수 없습니다.")
        return
    
    # 데이터셋에서 사용되는 클래스 목록 추출 (중복 제거 후 소문자로 정리)
    all_labels = [sample["label"].strip().lower() for sample in dataset]
    unique_labels = sorted(list(set(all_labels)))
    print("평가에 사용될 클래스:", unique_labels)
    
    # 텍스트 프롬프트 생성 (예: "a photo of a {class}")
    class_prompts = [f"a photo of a {label}" for label in unique_labels]
    
    # 전체 평가 결과를 저장할 딕셔너리
    overall_metrics = {}
    
    # 모델별 평가 수행
    for model_name in clip_models:
        print("\n============================")
        print(f"모델: {model_name}")
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        
        # 해당 모델에 대해 예측 결과 (유사도별 true/pred 값)를 얻음
        results = evaluate_model(model, preprocess, dataset, class_prompts, similarity_funcs, device)
        
        # 유사도별 평가지표 계산 후 출력
        overall_metrics[model_name] = {}
        for sim_name, res in results.items():
            metrics = compute_metrics(res["true"], res["pred"], labels=unique_labels)
            overall_metrics[model_name][sim_name] = metrics
            
            # 지표 출력
            print(f"[{sim_name}] Accuracy: {metrics['accuracy']*100:.2f}%, "
                  f"Precision: {metrics['precision']*100:.2f}%, "
                  f"Recall: {metrics['recall']*100:.2f}%, "
                  f"F1: {metrics['f1']*100:.2f}%")
    
    # 시각화: 모델 및 유사도 방식별 평가 지표를 bar chart로 표시
    visualize_metrics(overall_metrics)
    
if __name__ == "__main__":
    main()
```

---

## 코드 설명

1. **데이터셋 로드 및 전처리:**  
   - `load_dataset` 함수는 JSON 파일에서 각 이미지의 경로와 레이블을 읽어옵니다.  
   - 데이터셋의 레이블은 소문자로 정리하여 고유 클래스 목록(`unique_labels`)을 생성합니다.

2. **유사도 측정 함수:**  
   - 세 가지 함수(코사인, 내적, 유클리드 기반)를 통해 이미지 임베딩과 텍스트 임베딩 간의 유사도를 계산합니다.

3. **평가 함수 (`evaluate_model`):**  
   - 각 이미지에 대해 모델의 이미지 임베딩을 계산하고, 각 유사도 함수별로 가장 높은 점수를 가진 프롬프트(클래스)를 예측합니다.  
   - 예측값과 실제값을 각 유사도 방식별로 리스트에 저장합니다.

4. **평가지표 계산 (`compute_metrics`):**  
   - scikit-learn의 함수를 사용하여 Accuracy, Precision, Recall, F1 값을 계산합니다.  
   - 다중 클래스 평가의 경우 `average='macro'` 옵션을 사용합니다.

5. **시각화 (`visualize_metrics`):**  
   - pandas DataFrame으로 평가 결과를 정리한 후, seaborn의 barplot을 이용해 모델별/유사도 방식별 평가 지표(Accuracy, Precision, Recall, F1)를 subplot으로 시각화합니다.

6. **메인 실행:**  
   - 지정된 JSON 데이터셋과 모델 목록에 대해 평가를 진행한 후, 각 평가 지표 값을 터미널에 출력하고 최종적으로 시각화합니다.

이 코드를 실행하기 전에 [CLIP GitHub 저장소](https://github.com/openai/CLIP), PyTorch, scikit-learn, pandas, matplotlib, seaborn 등이 설치되어 있어야 합니다.  
혹시 추가적인 설명이나 수정이 필요하시면 언제든지 문의해 주세요.


