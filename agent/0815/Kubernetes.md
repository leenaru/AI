아래에 **요약 → 자세한 설명** 순서로 정리했습니다. (모든 핵심 개념은 최신 공식 문서에 근거하며, 각 단락 끝의 링크를 눌러 바로 확인하실 수 있습니다.)

---

## 요약

* \*\*쿠버네티스(Kubernetes)\*\*는 컨테이너화된 애플리케이션의 **배포·스케일링·자원관리**를 자동화하는 **오픈소스 오케스트레이션 플랫폼**입니다. 컨트롤 플레인(API 서버, 스케줄러 등)과 워커 노드(파드 실행)로 구성되며 **선언형(Declarative)** 방식으로 원하는 상태를 기술하면 컨트롤러가 실제 상태를 동기화합니다. ([Kubernetes][1])
* **핵심 오브젝트**: 파드(Pod), 디플로이먼트(Deployment), 서비스(Service), 인그레스(Ingress), 스테이트풀셋(StatefulSet), 데몬셋(DaemonSet), 잡/크론잡(Job/CronJob). 스토리지는 **PV/PVC/StorageClass**로 다룹니다. ([Kubernetes][2])
* **네트워킹/보안**: CNI 플러그인으로 파드 네트워킹을 제공하고, `kube-proxy`(혹은 Cilium eBPF 대체)를 통해 서비스 트래픽을 처리합니다. 보안은 **네임스페이스/RBAC/시크릿/네트워크폴리시/Pod Security Admission**으로 계층화합니다. ([Kubernetes][3], [docs.cilium.io][4])
* **오토스케일링/관측**: 워크로드 자동 확장은 **HPA**(수평), **VPA**(수직), **클러스터 오토스케일러**(노드)를 사용하고, 관측은 **Metrics Server/Prometheus Operator/OpenTelemetry Operator**로 표준화합니다. ([Kubernetes][5], [GitHub][6], [Kubernetes Sigs][7], [OpenTelemetry][8])
* **사용 방법**: 로컬은 **kind/minikube**로 빠르게 시작, `kubectl apply`로 매니페스트 배포, **Helm/Kustomize**로 설정 관리, 실제 프로덕션은 매니지드 서비스(EKS/GKE 등) 또는 온프렘에 배포합니다. ([KIND][9], [Minikube][10], [Kubernetes][11], [헬름][12])

---

# 1) 쿠버네티스란 무엇인가

쿠버네티스는 컨테이너 워크로드/서비스를 **선언형 구성과 자동화**로 운영하도록 설계된 플랫폼입니다. CNCF가 호스팅하며, 어디서나 동일하게 실행되는 **이식성·확장성**을 제공합니다. ([Kubernetes][1])

### 아키텍처 한눈에

* **컨트롤 플레인**: `kube-apiserver`, `etcd`(상태 저장), `kube-scheduler`(스케줄링), `kube-controller-manager`(조정 루프).
* **노드 컴포넌트**: `kubelet`(파드 생성/관리), `kube-proxy`(서비스 트래픽). ([Kubernetes][13])

---

# 2) 핵심 리소스와 개념

## 워크로드

* **Pod**: 쿠버네티스의 최소 배포 단위(한 개 이상 컨테이너의 그룹). ([Kubernetes][2])
* **Deployment**: 무상태 앱의 선언적 롤링 업데이트/롤백 관리. ([Kubernetes][14])
* **StatefulSet**: **고유 ID/순서/스토리지**를 보장하는 상태ful 워크로드. ([Kubernetes][15])
* **DaemonSet**: 각(또는 일부) 노드마다 파드 1개씩 실행(예: 로그/에이전트). ([Kubernetes][16])
* **Job/CronJob**: 일회성/주기성 배치 작업. ([Kubernetes][17])

## 서비스/인그레스

* **Service**: 파드 집합에 **안정적 가상 IP** 제공, 로드밸런싱.
* **Ingress**: HTTP(S) 레벨에서 **호스트/경로 기반** 라우팅·TLS 종료. (컨트롤러 필요: NGINX, Traefik 등) ([Kubernetes][18])

## 스토리지

* **Volume**: 파드 내 컨테이너 간 공유 파일시스템.
* **PV/PVC**: 클러스터 수준의 스토리지(PV)와 사용자 요청(PVC) 바인딩, **StorageClass**로 동적 프로비저닝. ([Kubernetes][19])

### 간단 예시(YAML)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: web, labels: { app: web } }
spec:
  replicas: 3
  selector: { matchLabels: { app: web } }
  template:
    metadata: { labels: { app: web } }
    spec:
      containers:
        - name: nginx
          image: nginx:1.27
          ports: [{containerPort: 80}]
---
apiVersion: v1
kind: Service
metadata: { name: web-svc }
spec:
  selector: { app: web }
  ports: [{ port: 80, targetPort: 80 }]
  type: ClusterIP
```

> `kubectl apply -f manifest.yaml` 로 배포합니다. ([Kubernetes][20])

---

# 3) 네트워킹과 서비스 디스커버리

* **CNI**: 파드 네트워킹은 컨테이너 런타임이 CNI 플러그인과 함께 구현(Calico, Cilium 등). ([Kubernetes][3])
* **kube-proxy**: 서비스 규칙을 각 노드에 반영(iptables/IPVS 등). 일부 환경에선 **Cilium eBPF**로 대체 가능(성능/가시성 이점). ([Kubernetes][21], [docs.cilium.io][4])

---

# 4) 보안 모델

* **네임스페이스**로 리소스 경계를 나누고, **RBAC**으로 세밀한 권한 제어. ([Kubernetes][22])
* \*\*시크릿(Secret)\*\*으로 민감정보 분리(파일/환경변수 주입). ([Kubernetes][23])
* \*\*네트워크폴리시(NetworkPolicy)\*\*로 파드 간 트래픽을 화이트리스트 방식으로 제어(지원 CNI 필요). ([Kubernetes][24])
* **Pod Security Admission**으로 네임스페이스 수준 보안 기준(Privileged/Baseline/Restricted) 강제. ([Kubernetes][25])
* **SecurityContext**로 루트 권한/Capabilities/SELinux 등 세부 제어. ([Kubernetes][26])

---

# 5) 스케일링과 가용성

* **HPA**: CPU/메모리/커스텀 지표로 파드 수 자동 조정. ([Kubernetes][5])
* **VPA**: 파드의 **요청/제한 자원값**을 자동 권장/조정(별도 컴포넌트). ([GitHub][6])
* **클러스터 오토스케일러**: 노드 풀 크기 자동 증감. ([Kubernetes][27])

---

# 6) 관측(Observability)

* **Metrics Server**: HPA에 필요한 리소스 지표 제공(`kubectl top`). ([Kubernetes Sigs][7])
* **Prometheus Operator**: Prometheus/Alertmanager를 CRD로 표준 운영. ([GitHub][28])
* **OpenTelemetry Operator**: 수집기/오토 인스트루먼테이션로 메트릭·로그·트레이스 파이프라인을 쿠버네티스식으로 관리. ([OpenTelemetry][8])

---

# 7) 실전 시작 가이드

## 로컬 개발: kind / minikube

* **kind**: Docker 안에 “컨테이너 노드”로 K8s 클러스터 구성.

  ```bash
  go install sigs.k8s.io/kind@v0.29.0
  kind create cluster --wait 3m
  kubectl get nodes
  ```

  (로컬 레지스트리/로드밸런서/오프라인 모드 등도 지원) ([KIND][9])

* **minikube**: 단일 노드 K8s를 한 명령으로 기동.

  ```bash
  minikube start
  kubectl apply -f manifest.yaml
  minikube addons enable ingress
  ```

  (Hello Minikube 튜토리얼로 배포/로그 확인) ([Minikube][10], [Kubernetes][29])

## 기본 `kubectl` 흐름

```bash
kubectl get ns,pods,svc,deploy -A
kubectl describe deploy web
kubectl logs deploy/web
kubectl rollout status deploy/web
kubectl scale deploy/web --replicas=5
```

필수 치트시트는 **kubectl Quick Reference**를 보시면 됩니다. ([Kubernetes][11])

## 스토리지(PV/PVC) 간단 예시

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata: { name: standard }
provisioner: kubernetes.io/no-provisioner  # 예시(동적 프로비저너 있으면 그 CSI 사용)
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: v1
kind: PersistentVolume
metadata: { name: pv-1 }
spec:
  capacity: { storage: 5Gi }
  accessModes: ["ReadWriteOnce"]
  storageClassName: standard
  hostPath: { path: /data/pv-1 }  # 데모용
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: web-pvc }
spec:
  accessModes: ["ReadWriteOnce"]
  resources: { requests: { storage: 5Gi } }
  storageClassName: standard
```

PVC를 파드에 마운트하는 절차는 공식 태스크 문서를 참고하세요. ([Kubernetes][30])

## HPA 예시(메트릭 서버 필요)

```bash
kubectl autoscale deploy web --min=3 --max=10 --cpu-percent=70
kubectl get hpa
```

자세한 워크스루는 HPA 문서를 참고하세요. ([Kubernetes][31])

## 인그레스 간단 예시(minikube에서 NGINX 인그레스 사용 시)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ing
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
    - host: web.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service: { name: web-svc, port: { number: 80 } }
```

(인그레스 컨트롤러를 별도로 설치/활성화해야 동작합니다.) ([Kubernetes][32])

---

# 8) 구성 관리: Helm vs Kustomize

* **Helm**: 차트로 패키징·버전·업그레이드/롤백 제공(패키지 매니저). 대형 앱 배포에 적합. ([헬름][33])
* **Kustomize**: 템플릿 없이 **오버레이/패치**로 환경별 차이를 관리(`kubectl apply -k`). GitOps(Argo CD)와 궁합이 좋습니다. ([Kubernetes][34])

---

# 9) 운영 베스트 프랙티스(요지)

* **RBAC 최소권한**·네임스페이스·리소스쿼터·네트워크폴리시 기본값으로 잠그기. ([Kubernetes][35])
* **Pod Security Admission**으로 베이스라인/리스트릭티드 적용. ([Kubernetes][25])
* **관측 스택**(Metrics Server + Prometheus Operator + OTEL)과 **경보** 필수. ([Kubernetes Sigs][7], [GitHub][28], [OpenTelemetry][8])
* **스토리지 클래스/백업**·**레디니스/라이브니스 프로브**·**헬스체크**·**롤링업데이트 전략**·**리소스 요청/제한** 설정.
* **오토스케일링**(HPA/VPA/클러스터 오토스케일러)로 비용/성능 균형. ([Kubernetes][5], [GitHub][6])

---

# 10) 배포 옵션 한눈에

* **로컬/개발**: kind(가볍고 재현성), minikube(애드온·학습 친화). ([KIND][9], [Minikube][10])
* **온프렘/베어메탈**: kubeadm으로 직접 부트스트랩 + CNI 선택(Calico/Cilium), 로드밸런서는 MetalLB 등 고려. ([Kubernetes][36])
* **매니지드**: EKS/GKE/OKE 등(컨트롤 플레인 관리·오토스케일링 통합·업데이트 간소화). ([AWS Documentation][37], [Google Cloud][38], [Oracle Docs][39])

---

## 마무리

원하시면 **실습용 레포 구조(Helm/Kustomize 둘 다)**, **관측/보안/스케일링**을 묶은 **샘플 클러스터 구성**(kind·minikube 기준)과 함께, **Grafana 대시보드 임포트용 JSON**까지 이어서 만들어 드리겠습니다. 어떤 환경(온프렘/클라우드)부터 시작할지 알려주시면 그에 맞춰 **단계별 체크리스트**도 같이 드릴게요.

[1]: https://kubernetes.io/docs/concepts/overview/?utm_source=chatgpt.com "Overview"
[2]: https://kubernetes.io/docs/concepts/workloads/pods/?utm_source=chatgpt.com "Pods"
[3]: https://kubernetes.io/docs/concepts/cluster-administration/networking/?utm_source=chatgpt.com "Cluster Networking"
[4]: https://docs.cilium.io/en/stable/network/kubernetes/kubeproxy-free.html?utm_source=chatgpt.com "Kubernetes Without kube-proxy"
[5]: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/?utm_source=chatgpt.com "Horizontal Pod Autoscaling"
[6]: https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler?utm_source=chatgpt.com "Kubernetes Vertical Pod Autoscaler (VPA)"
[7]: https://kubernetes-sigs.github.io/metrics-server/?utm_source=chatgpt.com "Kubernetes Metrics Server - GitHub Pages"
[8]: https://opentelemetry.io/docs/platforms/kubernetes/operator/?utm_source=chatgpt.com "OpenTelemetry Operator for Kubernetes"
[9]: https://kind.sigs.k8s.io/docs/user/quick-start/?utm_source=chatgpt.com "Quick Start - kind - Kubernetes"
[10]: https://minikube.sigs.k8s.io/docs/start/?utm_source=chatgpt.com "minikube start - Kubernetes"
[11]: https://kubernetes.io/docs/reference/kubectl/quick-reference/?utm_source=chatgpt.com "kubectl Quick Reference"
[12]: https://helm.sh/docs/?utm_source=chatgpt.com "Helm | Docs"
[13]: https://kubernetes.io/docs/concepts/architecture/?utm_source=chatgpt.com "Cluster Architecture"
[14]: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/?utm_source=chatgpt.com "Deployments"
[15]: https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/?utm_source=chatgpt.com "StatefulSets"
[16]: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/?utm_source=chatgpt.com "DaemonSet"
[17]: https://kubernetes.io/docs/concepts/workloads/controllers/job/?utm_source=chatgpt.com "Jobs"
[18]: https://kubernetes.io/docs/concepts/services-networking/service/?utm_source=chatgpt.com "Service"
[19]: https://kubernetes.io/docs/concepts/storage/volumes/?utm_source=chatgpt.com "Volumes"
[20]: https://kubernetes.io/docs/tutorials/kubernetes-basics/?utm_source=chatgpt.com "Learn Kubernetes Basics"
[21]: https://kubernetes.io/docs/reference/command-line-tools-reference/kube-proxy/?utm_source=chatgpt.com "kube-proxy"
[22]: https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/?utm_source=chatgpt.com "Namespaces"
[23]: https://kubernetes.io/docs/concepts/configuration/secret/?utm_source=chatgpt.com "Secrets"
[24]: https://kubernetes.io/docs/concepts/services-networking/network-policies/?utm_source=chatgpt.com "Network Policies"
[25]: https://kubernetes.io/docs/concepts/security/pod-security-admission/?utm_source=chatgpt.com "Pod Security Admission"
[26]: https://kubernetes.io/docs/tasks/configure-pod-container/security-context/?utm_source=chatgpt.com "Configure a Security Context for a Pod or Container"
[27]: https://kubernetes.io/docs/concepts/cluster-administration/node-autoscaling/?utm_source=chatgpt.com "Node Autoscaling"
[28]: https://github.com/prometheus-operator/prometheus-operator?utm_source=chatgpt.com "Prometheus Operator creates/configures/manages ..."
[29]: https://kubernetes.io/docs/tutorials/hello-minikube/?utm_source=chatgpt.com "Hello Minikube"
[30]: https://kubernetes.io/docs/tasks/configure-pod-container/configure-persistent-volume-storage/?utm_source=chatgpt.com "Configure a Pod to Use a PersistentVolume for Storage"
[31]: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/?utm_source=chatgpt.com "HorizontalPodAutoscaler Walkthrough"
[32]: https://kubernetes.io/docs/concepts/services-networking/ingress/?utm_source=chatgpt.com "Ingress"
[33]: https://helm.sh/?utm_source=chatgpt.com "Helm charts"
[34]: https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/?utm_source=chatgpt.com "Declarative Management of Kubernetes Objects Using ..."
[35]: https://kubernetes.io/docs/concepts/security/rbac-good-practices/?utm_source=chatgpt.com "Role Based Access Control Good Practices"
[36]: https://kubernetes.io/docs/reference/setup-tools/kubeadm/kubeadm-init/?utm_source=chatgpt.com "kubeadm init"
[37]: https://docs.aws.amazon.com/eks/latest/userguide/eks-architecture.html?utm_source=chatgpt.com "Amazon EKS architecture"
[38]: https://cloud.google.com/kubernetes-engine/docs?utm_source=chatgpt.com "Google Kubernetes Engine documentation"
[39]: https://docs.oracle.com/iaas/Content/ContEng/Concepts/contengoverview.htm?utm_source=chatgpt.com "Overview of Kubernetes Engine (OKE)"
