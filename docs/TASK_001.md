# TASK.md

## 작업 식별자

- **Task ID:** T001
- **작업명:** HDZoo-compatible nonlinear HDC core 및 UCI-HAR smoke 실행
- **기준 문서:** `RESEARCH_SPEC.md`
- **상태:** Ready

## 1. 현재 수행할 단일 작업

PyTorch로 HDZoo의 non-binary nonlinear encoding과 동등한 HDC core를 구현하고,
실제 UCI-HAR 데이터의 3개 subject를 이용해 로컬 prototype 학습 smoke 실행을 완성한다.

최종적으로 다음 명령들이 실행되어야 한다.

```bash
python -m horu_artifact prepare-data ucihar --data-root data
python -m horu_artifact smoke \
  --config configs/smoke_ucihar.yaml \
  --data-root data \
  --output results/smoke_ucihar \
  --device cpu
```

`prepare-data`는 공식 UCI 저장소에서 데이터를 받아 로컬 cache를 만든다.
그 이후 `smoke`는 네트워크 연결 없이 실행되어야 한다.

Smoke 실행은 federated aggregation을 수행하지 않는다. 동일한 server projection
matrix를 3개 client가 공유하되, 각 client는 자기 train split으로 독립적인 prototype
classifier를 초기화하고 error-driven update를 수행한다.

처리 순서는 다음과 같다.

1. UCI-HAR 원본 train/test 파일을 읽고 subject 정보를 보존하여 결합
2. 지정된 3개 subject별 deterministic stratified train/test split 생성
3. 데이터 전처리 단계에서 sample-wise L2 normalization
4. 서버에서 고정 nonlinear projection matrix 생성
5. 모든 client에 동일한 projection matrix 전달
6. nonlinear HD encoding
7. client별 class prototype memory 초기화
8. cosine-similarity prediction
9. 오분류 sample에 대한 push–pull update
10. client별 및 전체 accuracy와 실행 정보를 JSON으로 저장

이 작업의 목적은 이후 FedHDC, HyperFeel, HoRU가 함께 사용할 encoder,
prototype 연산, device 처리, 실제 데이터 입출력 구조를 확정하는 것이다.

## 2. 변경 범위

### 2.1 생성 또는 변경할 파일

```text
pyproject.toml
src/horu_artifact/__init__.py
src/horu_artifact/__main__.py
src/horu_artifact/cli.py
src/horu_artifact/config.py
src/horu_artifact/runtime.py
src/horu_artifact/hdc/__init__.py
src/horu_artifact/hdc/encoder.py
src/horu_artifact/hdc/prototype.py
src/horu_artifact/datasets/__init__.py
src/horu_artifact/datasets/ucihar.py
src/horu_artifact/smoke.py
configs/smoke_ucihar.yaml
data/README.md
tests/test_encoder.py
tests/test_prototype.py
tests/test_ucihar.py
tests/test_smoke_cli.py
```

현재 저장소에 동등한 구조가 있으면 중복 파일을 만들지 않고 기존 구조를 사용한다.

### 2.2 구현 기술과 device 정책

- Python 3.11
- PyTorch tensor 연산 사용
- 기본 dtype: `torch.float32`
- 기본 device: `cpu`
- 허용 device 값: `cpu`, `cuda`, `auto`
- `auto`는 CUDA 사용 가능 시 `cuda`, 아니면 `cpu`
- 테스트 suite는 CUDA 없이 전부 통과해야 함
- CUDA가 있으면 별도의 조건부 parity test를 실행
- projection matrix는 CPU의 명시적 `torch.Generator`로 생성한 뒤 선택한 device로 복사
- 전역 RNG에 의존하지 않음
- 공개 함수에 type hint와 docstring 작성
- 빈 입력, shape 불일치, 잘못된 label 및 device에는 명시적 예외 발생
- 입력 tensor와 model tensor의 device가 다르면 암묵적으로 복사하지 말고 오류 발생

NumPy는 텍스트 데이터 parsing 등 보조 작업에만 허용한다. Encoder, prototype
초기화, prediction, update는 PyTorch로 구현한다.

CPU 실행은 재현성 기준 경로이다. 이후 전체 accuracy 실험은 동일 구현에
`--device cuda`를 지정하여 실행할 수 있어야 한다. CPU와 CUDA 결과는 부동소수점
차이 때문에 bitwise 동일성을 요구하지 않고 tolerance 내 수치 일치만 요구한다.

### 2.3 HDZoo reference와 Encoder 고정 정의

다음 구현을 알고리즘 reference로 고정한다.

```text
Repository: CELL-POSTECH/HDZoo
Commit: 78046e5517de3e6bb4b9cab1cc73535a148b1376
File: hdzoo/core/encoder.py
Function: encode_nonlinear
Mode: nonbinarize=True
```

HDZoo의 해당 경로는 Gaussian base를 만든 뒤 matrix multiplication과 cosine을
적용하고, `nonbinarize=True`일 때 hard sign을 적용하지 않는다.

입력 feature 수를 `F`, HD dimension을 `D`라 한다. 서버는 seed로부터 다음
projection matrix를 한 번 생성한다.

```text
E ∈ R^(F×D), E[f,d] ~ Normal(0,1)
```

HDZoo의 base 생성 순서를 명확히 따르기 위해 CPU에서 다음 의미로 생성한다.

```python
raw = torch.empty((D, F), dtype=torch.float32, device="cpu")
raw.normal_(mean=0.0, std=1.0, generator=generator)
E = raw.transpose(0, 1).contiguous()
```

입력 batch `X ∈ R^(N×F)`의 encoding은 정확히 다음과 같다.

```text
H = cos(XE), H ∈ R^(N×D)
```

필수 정책:

- `sign`, hard sign, binary 또는 ternary 변환을 적용하지 않음
- random phase 또는 bias를 추가하지 않음
- projection matrix를 학습하지 않음
- client별 projection matrix를 만들지 않음
- encoder 내부에서 input 또는 output을 normalize하지 않음
- UCI-HAR의 sample-wise L2 normalization은 dataset preprocessing에서 수행
- encoding은 batch size와 무관하게 같은 결과를 생성

HDZoo package를 runtime dependency로 추가하지 않는다. 위 commit의 수학적 동작을
독립적으로 구현하고 reference parity test로 검증한다.

### 2.4 Prototype memory

class 수를 `K`라 할 때 각 client의 prototype memory는 다음과 같다.

```text
M_i ∈ R^(K×D)
```

각 class prototype은 해당 client train split의 encoded sample 평균을 계산한 뒤
L2 row normalization한다. sample이 없는 class의 행은 영벡터로 유지한다.

예측은 encoded query와 각 prototype의 cosine similarity로 수행한다. 영벡터
prototype의 score는 `-inf`로 처리하여 선택되지 않게 한다. 모든 prototype이
영벡터인 잘못된 상태에서는 예외를 발생시킨다.

### 2.5 Error-driven update

sample `(h, y)`에 대해 `ŷ != y`이면 다음 update를 수행한다.

```text
M_i[y] ← M_i[y] + ηh
M_i[ŷ] ← M_i[ŷ] - ηh
```

정분류 sample은 update하지 않는다. `η`와 local epoch 수는 config에서 읽는다.
이번 작업에서는 sample update 직후 prototype을 재정규화하지 않는다.

학습 sample 순서는 seed가 지정된 `torch.Generator`로 epoch마다 결정한다.
같은 device, seed, 데이터, config에서는 같은 순서와 결과를 생성해야 한다.

### 2.6 UCI-HAR 데이터 처리

공식 UCI Machine Learning Repository의 Dataset ID 240을 사용한다.

고정 metadata:

```text
Dataset: Human Activity Recognition Using Smartphones
DOI: 10.24432/C54S4K
License: CC BY 4.0
Expected features: 561
Expected classes: 6
Expected subjects: 30
Expected total samples: 10299
```

HDZoo가 제공하는 Google Drive dataset bundle이나 `.choir_dat` 파일은 사용하지
않는다. 공식 UCI archive를 직접 사용하고 다운로드 URL, 파일 크기, SHA-256,
다운로드 시각을 `data/ucihar/manifest.json`에 기록한다.

Loader는 원본의 다음 정보를 읽는다.

```text
train/X_train.txt
train/y_train.txt
train/subject_train.txt
test/X_test.txt
test/y_test.txt
test/subject_test.txt
```

원본 train/test를 합쳐 30개 subject를 복원한다. Smoke에서는 config에 지정된
subject ID `[1, 2, 3]`만 선택한다. 각 subject 내부에서 class별로 deterministic
70/30 train/test split을 만든다.

전처리:

- label을 내부 표현 `0..5`로 변환
- 각 sample을 L2 normalize
- split 이후에도 subject ID와 원본 row index 보존
- 누락값, feature 수, class 수, subject 수 검증
- cache 파일에는 source manifest와 preprocessing config를 포함

`prepare-data`만 네트워크를 사용할 수 있다. `smoke`는 cache가 없으면 다운로드를
시도하지 말고 준비 명령을 안내하며 non-zero exit code로 종료한다.

### 2.7 Smoke config와 출력

`configs/smoke_ucihar.yaml`의 기본값은 다음 의미를 가져야 한다.

```yaml
dataset: ucihar
subject_ids: [1, 2, 3]
test_ratio: 0.3
seed: 0
hd_dim: 256
learning_rate: 0.035
local_epochs: 1
batch_size: 128
device: cpu
```

`results/smoke_ucihar/result.json`에는 최소 다음 필드를 저장한다.

```json
{
  "status": "pass",
  "dataset": "ucihar",
  "dataset_doi": "10.24432/C54S4K",
  "dataset_sha256": "",
  "seed": 0,
  "device": "cpu",
  "torch_version": "",
  "num_clients": 3,
  "client_ids": [1, 2, 3],
  "num_classes": 6,
  "input_dim": 561,
  "hd_dim": 256,
  "projection_shape": [561, 256],
  "projection_sha256": "",
  "initial_mean_accuracy": 0.0,
  "final_mean_accuracy": 0.0,
  "num_updates": 0,
  "per_client": [],
  "output_files": []
}
```

추가 산출물:

```text
results/smoke_ucihar/config.resolved.yaml
results/smoke_ucihar/environment.json
results/smoke_ucihar/client_metrics.csv
```

기존 output directory가 비어 있지 않으면 `--overwrite` 없이는 실패해야 한다.
실패 시 `status: pass` 결과를 남기지 않는다.

### 2.8 명시적 비범위

이번 작업에서는 다음을 구현하지 않는다.

- client 간 model aggregation
- shared/common/global/personal basis
- eigendecomposition 또는 SVD
- FedHDC, HyperFeel, HoRU
- 25-round federated experiment
- 나머지 5개 실제 데이터셋
- Docker
- latency/energy benchmark
- communication payload 비교
- FedAvg 또는 DFL
- figure/table 생성
- 논문 수치에 맞춘 tuning
- HDZoo 전체 코드를 복사하거나 dependency로 설치
- `RESEARCH_SPEC.md` 변경

placeholder, mock algorithm, hard-coded accuracy 및 precomputed prediction을 금지한다.

## 3. 테스트

### 3.1 Encoder unit test

`tests/test_encoder.py`

1. 동일 seed, `F`, `D`에서 동일 projection matrix를 생성한다.
2. 다른 seed에서는 다른 matrix를 생성한다.
3. projection shape은 `(F, D)`, dtype은 `torch.float32`, 생성 위치는 CPU이다.
4. projection 생성 순서가 고정된 HDZoo reference 구현과 정확히 일치한다.
5. encoder 출력이 `torch.cos(X @ E)`와 tolerance 내에서 일치한다.
6. sign 또는 binarization이 적용되지 않았음을 음수·0이 아닌 연속값으로 확인한다.
7. 같은 projection을 받은 두 client의 encoding이 일치한다.
8. batch size를 변경해도 결과가 일치한다.
9. feature dimension 또는 device가 불일치하면 예외가 발생한다.
10. CUDA가 있으면 동일 CPU projection을 복사해 CPU/CUDA output이 tolerance 내에서 일치한다.

### 3.2 Prototype unit test

`tests/test_prototype.py`

1. prototype shape은 `(K, D)`이다.
2. sample이 존재하는 class 행의 L2 norm은 1이다.
3. sample이 없는 class 행은 영벡터이다.
4. cosine prediction이 수작업 계산과 일치한다.
5. 영벡터 prototype이 선택되지 않는다.
6. 모든 prototype이 영벡터이면 예외가 발생한다.
7. 오분류 시 true/predicted class 두 행만 변경된다.
8. 정분류 시 어느 행도 변경되지 않는다.
9. update 직후 자동 재정규화가 수행되지 않는다.
10. 잘못된 label, 빈 batch 또는 device 불일치에 예외가 발생한다.

### 3.3 UCI-HAR loader test

`tests/test_ucihar.py`

테스트는 네트워크를 사용하지 않고 임시 directory에 만든 최소 형식 fixture를 사용한다.

1. 필수 파일 누락 시 명확한 오류가 발생한다.
2. train/test 원본을 합친 뒤 subject ID가 보존된다.
3. label이 `0..5`로 변환된다.
4. sample-wise L2 norm이 1이다.
5. 동일 seed에서 subject별 split index가 동일하다.
6. train/test index가 겹치지 않는다.
7. class가 충분한 경우 stratified 70/30 비율이 유지된다.
8. feature 수, label 범위 및 subject 범위 검증이 동작한다.
9. cache에서 다시 읽은 tensor와 최초 tensor가 일치한다.
10. manifest에 source URL과 SHA-256이 기록된다.

### 3.4 CLI integration test

`tests/test_smoke_cli.py`

CI에서는 네트워크와 전체 UCI-HAR를 요구하지 않도록 local fixture cache를 사용한다.

1. `prepare-data`가 이미 준비된 cache를 검증할 수 있다.
2. cache가 없을 때 `smoke`가 다운로드하지 않고 실패한다.
3. 임시 output directory에서 smoke 명령이 exit code 0으로 종료된다.
4. JSON, resolved config, environment, client CSV가 생성된다.
5. 결과의 dataset, client ID, feature 수, class 수, projection shape이 config와 일치한다.
6. 모든 accuracy가 `[0,1]` 범위에 있다.
7. `num_updates > 0`이다.
8. 같은 CPU config로 두 번 실행했을 때 projection hash, split, update 수와 metric이 동일하다.
9. 세 client의 projection hash가 하나로 동일하다.
10. 기존 non-empty output directory는 `--overwrite` 없이 실패한다.

### 3.5 실제 데이터 acceptance test

개발 환경에서 다음을 실행한다.

```bash
python -m pip install -e ".[test]"
pytest -q

python -m horu_artifact prepare-data ucihar --data-root data

python -m horu_artifact smoke \
  --config configs/smoke_ucihar.yaml \
  --data-root data \
  --output results/smoke_ucihar \
  --device cpu
```

CUDA 환경에서는 추가로 실행한다.

```bash
python -m horu_artifact smoke \
  --config configs/smoke_ucihar.yaml \
  --data-root data \
  --output results/smoke_ucihar_cuda \
  --device cuda
```

CPU와 CUDA의 projection hash, split index, sample 수는 같아야 한다.
동일 입력에 대한 encoded tensor는 `rtol=1e-4`, `atol=1e-5` 이내 일치를 검사한다.
최종 prediction과 accuracy는 부동소수점 경계에서 달라질 수 있으므로 CPU 결과를
재현성 기준값으로 사용하고, CUDA mismatch 수와 accuracy 차이를 `RESULTS.md`에 기록한다.

## 4. 완료 조건

다음 조건을 모두 만족해야 완료이다.

- 실제 구현에 stub 또는 미해결 `TODO`가 없다.
- `pytest -q`가 CPU-only 환경에서 모두 통과한다.
- 공식 UCI-HAR 다운로드와 cache 생성이 성공한다.
- UCI-HAR metadata가 10,299 samples, 561 features, 6 classes, 30 subjects로 검증된다.
- 3개 subject의 deterministic local split으로 smoke 실행이 성공한다.
- 모든 client가 동일 projection hash를 사용한다.
- encoder가 HDZoo nonlinear non-binary 경로와 동등한 `cos(XE)`를 구현한다.
- sign, binary, random phase 및 encoder 내부 normalization이 없다.
- prototype 초기화, cosine prediction 및 push–pull update가 unit test로 검증된다.
- `--device cpu`, `--device auto`가 동작한다.
- CUDA 환경에서는 `--device cuda`가 동작한다.
- 결과 JSON, resolved config, environment 및 client CSV가 생성된다.
- `smoke` 실행 중에는 네트워크를 사용하지 않는다.
- 이번 작업의 비범위 코드를 선행 구현하지 않는다.
- 실행 명령, 테스트 결과, 생성 파일, CPU/CUDA 차이, 실패 및 미확인 사항을
  `RESULTS.md`에 기록한다.
