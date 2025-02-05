# 📌 **코드 분석 및 설명 (추가된 파일 포함)**

이번에 업로드된 추가 파일들은 MobileCLIP 학습 및 데이터 증강/강화 관련 설정과 실행 스크립트를 다룹니다.  
각 파일의 역할과 세부 내용을 설명드리겠습니다.

---

## **📂 파일 개요**
| 파일명                 | 주요 역할                                                                                   |
|------------------------|--------------------------------------------------------------------------------------------|
| `run_datacompdr1b.sh`  | DataCompDR-1B 데이터를 사용한 MobileCLIP 학습 실행 스크립트                                   |
| `run_datacompdr12m.sh` | DataCompDR-12M 데이터를 사용한 MobileCLIP 학습 실행 스크립트                                  |
| `run_datacomp12m.sh`   | DataComp-12M 데이터를 사용한 MobileCLIP 학습 실행 스크립트                                    |
| `datacompdr1b.json`    | DataCompDR-1B 데이터셋을 위한 학습 강화 설정 파일                                            |
| `datacompdr12m.json`   | DataCompDR-12M 데이터셋을 위한 학습 강화 설정 파일                                           |

---

# **1️⃣ `run_datacompdr1b.sh` 분석**
📌 **역할:**  
- **DataCompDR-1B** 데이터셋을 기반으로 MobileCLIP 학습을 실행하는 스크립트  
- 분산 학습(`torchrun`)을 사용하여 대규모 데이터셋을 처리

### 🔹 **핵심 코드**
```bash
num_gpus=8
num_nodes=16
global_batch_size=$((2**13*8))
num_seen_samples=$((200*1000*global_batch_size))
exp_name="mobileclipb_datacompdr1b_s13b_$(date +%Y-%m-%d_%H-%M-%S)"
data="DataCompDR-1B/{{00..64}/{00000000..00000999},65/{00000000..00000535}}.tar"
```
- **분산 환경 설정:** 8개의 GPU와 16개의 노드를 사용하여 학습  
- **데이터셋 구성:** `DataCompDR-1B` 데이터셋의 특정 샤드(.tar 파일)를 로드  
- **전역 배치 크기:** 65,536 (`8192 * 8`)  

```bash
torchrun --nproc_per_node $num_gpus --nnodes $num_nodes --node_rank $ROLE_RANK \
    --train-data "$data" \
    --dataset-reinforcement-config configs/datacompdr1b.json
```
- **학습 실행:** `torchrun`을 통해 MobileCLIP 모델 학습 실행  
- `datacompdr1b.json`에서 데이터 증강 및 학습 강화 설정을 로드  

---

# **2️⃣ `run_datacompdr12m.sh` 분석**
📌 **역할:**  
- **DataCompDR-12M** 데이터셋을 기반으로 MobileCLIP 학습을 실행하는 스크립트  
- `run_datacompdr1b.sh`와 유사하지만, 데이터셋 크기와 노드 설정이 다름

### 🔹 **핵심 코드**
```bash
num_gpus=8
num_nodes=4
global_batch_size=8192
num_seen_samples=$((30*1000*global_batch_size))
exp_name="mobileclipb_datacompdr12m_s30m_$(date +%Y-%m-%d_%H-%M-%S)"
data="DataCompDR-12M/shards/{00000000..00001023}.tar"
```
- **데이터셋 크기:** DataCompDR-12M 데이터셋의 1,024개 샤드를 처리  
- **분산 환경:** 8개의 GPU와 4개의 노드를 사용  

```bash
torchrun --train-data "$data" \
    --dataset-reinforcement-config configs/datacompdr12m.json
```
- **학습 강화 설정:** `datacompdr12m.json` 파일을 사용  

---

# **3️⃣ `run_datacomp12m.sh` 분석**
📌 **역할:**  
- **DataComp-12M** 데이터셋을 사용한 MobileCLIP 학습 스크립트  
- 단일 노드에서 실행

### 🔹 **핵심 코드**
```bash
num_gpus=8
num_nodes=1
global_batch_size=8192
data="DataCompDR-12M/shards/{00000000..00001023}.tar"
```
- 단일 노드와 8개의 GPU를 사용  
- 학습 데이터는 DataComp-12M의 샤드로 구성  

---

# **4️⃣ `datacompdr1b.json` 분석**
📌 **역할:**  
- **DataCompDR-1B 데이터셋**에 대한 학습 강화 설정 파일

### 🔹 **핵심 설정**
```json
{
  "model_name": "ViT-L-14,ViT-L-14",
  "num_samples": 10,
  "pretrained": "openai,datacomp_xl_s13b_b90k",
  "reinforce": {
    "image_augmentation": {
      "normalize": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711]
      },
      "rand_augment": {
        "enable": true,
        "p": 1.0
      },
      "random_resized_crop": {
        "interpolation": "bicubic",
        "size": 224
      }
    }
  }
}
```
- **사전 학습된 모델:** OpenAI의 `ViT-L-14`  
- **데이터 증강:**  
  - **랜덤 증강(`rand_augment`)** 활성화  
  - **이미지 크기 조정(`random_resized_crop`)**: 크기 224x224, `bicubic` 보간법 사용  

---

# **5️⃣ `datacompdr12m.json` 분석**
📌 **역할:**  
- **DataCompDR-12M 데이터셋**에 대한 학습 강화 설정 파일  
- `datacompdr1b.json`과 동일한 구조를 가지며, 샘플 수가 다름

### 🔹 **주요 차이점**
```json
"num_samples": 30
```
- **샘플 크기:** 30만 개로 설정  

---

# **📌 최종 정리**
| 파일명                 | 주요 역할                                                                                   |
|------------------------|--------------------------------------------------------------------------------------------|
| `run_datacompdr1b.sh`  | DataCompDR-1B 데이터셋을 사용하여 MobileCLIP 학습 실행                                       |
| `run_datacompdr12m.sh` | DataCompDR-12M 데이터셋을 사용하여 MobileCLIP 학습 실행                                      |
| `run_datacomp12m.sh`   | DataComp-12M 데이터셋을 단일 노드에서 MobileCLIP 학습 실행                                   |
| `datacompdr1b.json`    | DataCompDR-1B 데이터셋용 학습 강화 설정                                                     |
| `datacompdr12m.json`   | DataCompDR-12M 데이터셋용 학습 강화 설정                                                    |

---

MobileCLIP의 **분산 학습**, **데이터셋 강화**, **대규모 데이터 처리**를 위한 구조를 잘 나타냅니다.  
추가 질문이 있다면 언제든 말씀해주세요! 😊
