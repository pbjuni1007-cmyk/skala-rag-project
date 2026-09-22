# 고정 검색 실험 결과 — 2026-09-21

영어 주 평가의 사전 선택은 **e5-300-50**, 마지막 검증 후 유지할 설정은 **e5-300-50**다. 검증 통과 여부는 **True**다. 이 결론은 기술 논문 2편의 검색 실험에 한정하며 질문 정답의 의미는 아직 `human_review_pending`이다. 자동 점수와 사람이 검수한 품질을 구분한다.

- 선택 기록: [selection.json](../experiments/retrieval-experiments-20260921/selection.json)
- 마지막 판정: [decision.json](../experiments/retrieval-experiments-20260921/decision.json)
- 사전 고정 계약: [protocol.json](../experiments/retrieval-experiments-20260921/protocol.json), [manifest.json](../experiments/retrieval-experiments-20260921/manifest.json)
- 질문/정답: [retrieval-technical.json](../eval/retrieval-technical.json)
- 모델 설정: [retrieval-models.json](../eval/retrieval-models.json)
- 구현: [retrieval_experiments.py](../scripts/retrieval_experiments.py)

## 공통 평가 계약과 환경

실험일은 2026-09-21이다. KIVI 15페이지와 InfiniGen 18페이지의 기존 추출문을 사용했다. 기존 기술질문16개는 설정 선택용이며, 그 안의 원래 필수8개(K1–K4, I1–I4)를 질문별로 보호했다. 신규8개(H1–H8)는 Table 2, Figure 17, 표의 열 제목과 수치, 조건, 연속페이지 근거를 묻는 마지막 검증용이다. 영어가 주 평가이며 한국어 번역은 보조다. 두 언어로 늘어난32/16질의는 독립 질문32/16개가 아니다.

원문 페이지·인용문·offset·hash 및 데이터/모델/실행 스크립트 hash를 결과 확인 전에 고정했다. 선택 과정은 holdout 점수를 읽지 않는다. 선택 후에는 baseline과 선택 후보만 holdout으로 평가하고 재선택을 막는다. 현재 정답 상태는 사람검수전이며, 표·그림에서의 숫자 해석을 사람이 승인했다는 뜻이 아니다.

`완전근거`는 **등록된 정답구절의 완전 회수**를 의미한다. 질문의 모든 의미 요소를 충족한 정답이라는 판정은 아니다. 기존 K3 정답구절에는 ShareGPT·2bit·7페이지 문장 앞부분이, I1에는 partial weights·CPU memory가 충분히 포함되지 않은 한계가 독립 원문검증에서 제기됐다. 기존 구절과 판정 규칙은 결과 확인 후 바꾸지 않았다. 필요한 모든 등록 인용 구간이 검색 문맥의 정확한 문자 구간 합집합으로 100% 덮이는 경우다. 한 청크에 모두 있어야 한다는 조건은 없다. 조건 회수는 각 조건에 연결된 모든 span의 회수 여부 평균이다. 중복은 같은 원문 페이지·위치가 여러 청크에 겹치는 문자 비율이다. 이 지표는 답변의 사실성이나 생성 품질을 직접 평가하지 않는다.

`top5`는 상위5청크다. `budget2000`은 검색 순서대로 전체 청크를 넣되 결합 구분자까지 포함하여 **E5 tokenizer 기준 최대2000토큰**을 넘지 않는 문맥이다. 모든 설정에 같은 상한을 적용하며 실제 사용량이 꼭2000은 아니다. 잘라 넣지 못한 큰 청크를 건너뛰므로 상위 몇 청크만의 단순 접두구간과 다를 수 있다. `adjacent2000`은 1위 청크 다음에 같은 문서의 인접페이지 청크를 검색 순서대로 우선 넣은 별도 진단이며 운영 후보선택에 사용하지 않았다.

공통 본문 실험은 세 모델에 문자 단위로 같은 청크를 입력하고 각 모델의 접두사·특수토큰을 포함해 한도를 검사했다. 운영 청크 실험은 **각 모델 자신의 tokenizer**와 기존 `Corpus.build`의 offset/stride/마지막 겹침 청크 규칙을 사용한다. E5 baseline380/50의136청크는 실제 tokenizer로 기존 운영 본문과 동일함을 확인했다. 따라서 공통 본문 모델 차이와 운영 청킹 차이를 분리해서 읽어야 한다.

모델은 CPU, 정규화 dense embedding, `trust_remote_code=False`, 로컬 고정 snapshot으로 실행했다. 스레드는2개이며 M3는 batch1, 나머지는 batch16이다. 성능 측정 subprocess는 E5→BGE-small→M3 순차였다. warm query는 각 질의3회 실행했으며 검색 시간은 query encode+벡터 내적+정렬을 포함하고 문맥 조립은 제외한다. 색인 시간에는 모델 로딩을 포함하지 않는다. 색인 크기는 벡터 파일+청크 JSON이다. RSS는 프로세스 최대 resident memory(실제 메모리 상주량)이며 같은 모델 프로세스의 앞선 설정이 사용한 메모리도 포함할 수 있다.

환경: `macOS-26.6.2-arm64-arm-64bit`. Python `3.12.14`. 패키지: `{"torch": "2.14.0", "transformers": "5.17.0", "sentence-transformers": "5.7.0", "numpy": "2.5.3"}`. 모델별 로딩시간과 원시 질의시간은 개별 JSON에 보존했다. OS 배경 작업까지 통제한 마이크로벤치마크가 아니므로 작은 지연 차이를 일반화하지 않는다.

## [EXP-01] E5 청크9조합

목적은 chunk200/300/380 × overlap0/25/50의 검색 근거 회수와 비용 비교다. 아래 세 문맥 방식은 영어 선택16문항에 대한 별도 결과다.

| 설정 | top5 완전/조건 | 예산2000 완전/조건 | 인접페이지2000 완전/조건 |
| --- | ---: | ---: | ---: |
| [e5-200-0](../experiments/retrieval-experiments-20260921/results/e5-200-0-selection.json) | 68.75% / 71.88% | 75.00% / 78.12% | 43.75% / 50.00% |
| [e5-200-25](../experiments/retrieval-experiments-20260921/results/e5-200-25-selection.json) | 62.50% / 68.75% | 75.00% / 78.12% | 25.00% / 34.38% |
| [e5-200-50](../experiments/retrieval-experiments-20260921/results/e5-200-50-selection.json) | 75.00% / 78.12% | 81.25% / 84.38% | 37.50% / 43.75% |
| [e5-300-0](../experiments/retrieval-experiments-20260921/results/e5-300-0-selection.json) | 75.00% / 81.25% | 75.00% / 81.25% | 50.00% / 56.25% |
| [e5-300-25](../experiments/retrieval-experiments-20260921/results/e5-300-25-selection.json) | 75.00% / 75.00% | 75.00% / 75.00% | 31.25% / 37.50% |
| [e5-300-50](../experiments/retrieval-experiments-20260921/results/e5-300-50-selection.json) | 81.25% / 84.38% | 87.50% / 87.50% | 18.75% / 28.12% |
| [e5-380-0](../experiments/retrieval-experiments-20260921/results/e5-380-0-selection.json) | 75.00% / 75.00% | 75.00% / 75.00% | 68.75% / 71.88% |
| [e5-380-25](../experiments/retrieval-experiments-20260921/results/e5-380-25-selection.json) | 75.00% / 78.12% | 75.00% / 78.12% | 62.50% / 62.50% |
| [e5-380-50](../experiments/retrieval-experiments-20260921/results/e5-380-50-selection.json) | 68.75% / 71.88% | 75.00% / 78.12% | 68.75% / 68.75% |

실제 선택 기준인 예산2000의 중복 및 한국어 보조 결과:

| 설정·원시 JSON | 영어 완전근거 | 영어 조건 | 영어 중복 | 한국어 완전/조건(보조) |
| --- | ---: | ---: | ---: | ---: |
| [e5-200-0](../experiments/retrieval-experiments-20260921/results/e5-200-0-selection.json) | 75.00% | 78.12% | 0.00% | 62.50% / 68.75% |
| [e5-200-25](../experiments/retrieval-experiments-20260921/results/e5-200-25-selection.json) | 75.00% | 78.12% | 2.75% | 56.25% / 59.38% |
| [e5-200-50](../experiments/retrieval-experiments-20260921/results/e5-200-50-selection.json) | 81.25% | 84.38% | 5.97% | 56.25% / 62.50% |
| [e5-300-0](../experiments/retrieval-experiments-20260921/results/e5-300-0-selection.json) | 75.00% | 81.25% | 0.00% | 43.75% / 50.00% |
| [e5-300-25](../experiments/retrieval-experiments-20260921/results/e5-300-25-selection.json) | 75.00% | 75.00% | 1.94% | 37.50% / 40.62% |
| [e5-300-50](../experiments/retrieval-experiments-20260921/results/e5-300-50-selection.json) | 87.50% | 87.50% | 4.51% | 50.00% / 53.12% |
| [e5-380-0](../experiments/retrieval-experiments-20260921/results/e5-380-0-selection.json) | 75.00% | 75.00% | 0.00% | 56.25% / 59.38% |
| [e5-380-25](../experiments/retrieval-experiments-20260921/results/e5-380-25-selection.json) | 75.00% | 78.12% | 1.78% | 56.25% / 59.38% |
| [e5-380-50](../experiments/retrieval-experiments-20260921/results/e5-380-50-selection.json) | 75.00% | 78.12% | 4.31% | 50.00% / 53.12% |

| 설정 | 청크 | 색인 s | warm 검색 중앙값 ms | 색인 MiB | 프로세스 peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| e5-200-0 | 219 | 2.409 | 4.825 | 0.493 | 1445.7 |
| e5-200-25 | 245 | 2.773 | 4.881 | 0.553 | 1445.7 |
| e5-200-50 | 286 | 3.192 | 4.944 | 0.644 | 1445.7 |
| e5-300-0 | 152 | 2.781 | 4.852 | 0.385 | 1445.7 |
| e5-300-25 | 160 | 2.850 | 5.100 | 0.408 | 1445.7 |
| e5-300-50 | 175 | 3.182 | 4.849 | 0.446 | 1445.7 |
| e5-380-0 | 123 | 3.050 | 4.866 | 0.337 | 1445.7 |
| e5-380-25 | 126 | 3.257 | 4.841 | 0.350 | 1445.7 |
| e5-380-50 | 136 | 3.438 | 4.844 | 0.376 | 1445.7 |

평균 점수가 높더라도 원래 필수8문항 중 하나의 완전회수가 떨어지면 제외했다.

| 운영 후보 | 기존 필수8개 중 baseline 대비 하락한 영어 질문 | 선택 자격 |
| --- | --- | --- |
| e5-200-0 | K1, I2 | 제외 |
| e5-200-25 | 없음 | 통과 |
| e5-200-50 | I1, I2 | 제외 |
| e5-300-0 | K4, I2 | 제외 |
| e5-300-25 | I3 | 제외 |
| e5-300-50 | 없음 | 통과 |
| e5-380-0 | K1 | 제외 |
| e5-380-25 | K1 | 제외 |
| e5-380-50 | 없음 | 통과 |
| bge-small-380-50 | 없음 | 제외 |
| bge-m3-380-50 | 없음 | 통과 |

## [EXP-02] 동일 본문 모델 비교와 운영 청크 비교

공통 본문은 각 모델95청크이며 세 모델의 입력 본문이 동일하다. E5/BGE-small/M3의 실제 실험 입력 상한은 모두512다. 모델 공식 revision·prefix·공식 한도는 [공식 설정 기록](source/model-and-enterprise-evidence-20260921.md)에 있다. M3 공식8192 상한을 이 실험의 입력 길이로 사용하지 않았다.

| 설정·원시 JSON | 영어 완전근거 | 영어 조건 | 영어 중복 | 한국어 완전/조건(보조) |
| --- | ---: | ---: | ---: | ---: |
| [e5-common](../experiments/retrieval-experiments-20260921/results/e5-common-selection.json) | 75.00% | 75.00% | 0.00% | 56.25% / 56.25% |
| [bge-small-common](../experiments/retrieval-experiments-20260921/results/bge-small-common-selection.json) | 68.75% | 68.75% | 0.00% | 12.50% / 15.62% |
| [bge-m3-common](../experiments/retrieval-experiments-20260921/results/bge-m3-common-selection.json) | 87.50% | 87.50% | 0.00% | 68.75% / 71.88% |

| 설정 | top5 완전/조건 | 예산2000 완전/조건 | 인접페이지2000 완전/조건 |
| --- | ---: | ---: | ---: |
| [e5-common](../experiments/retrieval-experiments-20260921/results/e5-common-selection.json) | 81.25% / 81.25% | 75.00% / 75.00% | 56.25% / 56.25% |
| [bge-small-common](../experiments/retrieval-experiments-20260921/results/bge-small-common-selection.json) | 75.00% / 75.00% | 68.75% / 68.75% | 50.00% / 50.00% |
| [bge-m3-common](../experiments/retrieval-experiments-20260921/results/bge-m3-common-selection.json) | 87.50% / 87.50% | 87.50% / 87.50% | 50.00% / 50.00% |

| 설정 | 청크 | 색인 s | warm 검색 중앙값 ms | 색인 MiB | 프로세스 peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| e5-common | 95 | 3.841 | 4.865 | 0.292 | 1445.7 |
| bge-small-common | 95 | 3.656 | 4.883 | 0.292 | 1308.7 |
| bge-m3-common | 95 | 23.539 | 38.612 | 0.524 | 2176.8 |

공통 본문에서 공통 E5 대비 필수8개 영어 완전회수 비하락을 통과한 모델만 운영 후보로 넘긴다. 이후 운영 청크에서는 다시 baseline E5 380/50 대비 필수질문 비하락을 확인하고 전체 완전회수→조건회수→낮은 지연 순으로 하나를 고른다. 공통 본문의 평균 순위만으로 운영모델을 바꾸지 않는다.

| 공통 본문 모델 | 공통 E5 대비 필수질문 완전회수 하락 | 운영 후보 허용 |
| --- | --- | --- |
| e5-common | 없음 | 허용 |
| bge-small-common | I3, I4 | 제외 |
| bge-m3-common | 없음 | 허용 |

동일 명목380/50을 각 tokenizer로 자른 운영 청크 비교는 다음과 같다. E5 grid에서 선택된 별도 길이는 EXP-01과 최종 판정으로 연결된다.

| 설정·원시 JSON | 영어 완전근거 | 영어 조건 | 영어 중복 | 한국어 완전/조건(보조) |
| --- | ---: | ---: | ---: | ---: |
| [e5-380-50](../experiments/retrieval-experiments-20260921/results/e5-380-50-selection.json) | 75.00% | 78.12% | 4.31% | 50.00% / 53.12% |
| [bge-small-380-50](../experiments/retrieval-experiments-20260921/results/bge-small-380-50-selection.json) | 68.75% | 68.75% | 3.22% | 31.25% / 34.38% |
| [bge-m3-380-50](../experiments/retrieval-experiments-20260921/results/bge-m3-380-50-selection.json) | 81.25% | 81.25% | 3.43% | 81.25% / 81.25% |

| 설정 | top5 완전/조건 | 예산2000 완전/조건 | 인접페이지2000 완전/조건 |
| --- | ---: | ---: | ---: |
| [e5-380-50](../experiments/retrieval-experiments-20260921/results/e5-380-50-selection.json) | 68.75% / 71.88% | 75.00% / 78.12% | 68.75% / 68.75% |
| [bge-small-380-50](../experiments/retrieval-experiments-20260921/results/bge-small-380-50-selection.json) | 68.75% / 68.75% | 68.75% / 68.75% | 50.00% / 50.00% |
| [bge-m3-380-50](../experiments/retrieval-experiments-20260921/results/bge-m3-380-50-selection.json) | 75.00% / 78.12% | 81.25% / 81.25% | 31.25% / 34.38% |

| 설정 | 청크 | 색인 s | warm 검색 중앙값 ms | 색인 MiB | 프로세스 peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| e5-380-50 | 136 | 3.438 | 4.844 | 0.376 | 1445.7 |
| bge-small-380-50 | 132 | 3.411 | 4.987 | 0.370 | 1308.7 |
| bge-m3-380-50 | 136 | 25.594 | 38.011 | 0.708 | 2176.8 |

MiniLM은 이번 후보가 아니다. [과거 E5/MiniLM 실험](embedding-benchmark.md)은 당시 입력·질문·프로토콜의 기록이며 이 실험과 점수를 합치지 않았다.

## [EXP-03] M3 총입력1024 별도 탐색

M3의1024차원 출력 벡터와1024토큰 입력 실험은 다르다. 이 실험은 특수토큰까지 포함한 입력상한1024를 적용하도록 마지막 본문 token 경계를 줄였다. 운영512 실험과 별개이고 후보선택에 들어가지 않는다.

| 설정·원시 JSON | 영어 완전근거 | 영어 조건 | 영어 중복 | 한국어 완전/조건(보조) |
| --- | ---: | ---: | ---: | ---: |
| [bge-m3-1024-50](../experiments/retrieval-experiments-20260921/results/bge-m3-1024-50-selection.json) | 56.25% | 59.38% | 0.40% | 43.75% / 46.88% |

| 설정 | top5 완전/조건 | 예산2000 완전/조건 | 인접페이지2000 완전/조건 |
| --- | ---: | ---: | ---: |
| [bge-m3-1024-50](../experiments/retrieval-experiments-20260921/results/bge-m3-1024-50-selection.json) | 87.50% / 87.50% | 56.25% / 59.38% | 37.50% / 40.62% |

| 설정 | 청크 | 색인 s | warm 검색 중앙값 ms | 색인 MiB | 프로세스 peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge-m3-1024-50 | 61 | 25.099 | 38.101 | 0.391 | 2176.8 |

## [EXP-04] 사전 선택과 마지막 검증

선택용16문항만 보고 **e5-300-50**를 골랐다. 후보 목록·제외 결과·각 점수는 [selection.json](../experiments/retrieval-experiments-20260921/selection.json)에 있다. 검증 시작 시 selection hash를 [validation-lock.json](../experiments/retrieval-experiments-20260921/validation-lock.json)에 고정했다. 신규8개를 이용한 반복 튜닝은 하지 않았다.

| 설정 | 영어 완전근거 | 영어 조건 | 한국어 완전/조건(보조) |
| --- | ---: | ---: | ---: |
| [e5-380-50](../experiments/retrieval-experiments-20260921/results/e5-380-50-holdout.json) | 37.50% | 71.88% | 50.00% / 67.71% |
| [e5-300-50](../experiments/retrieval-experiments-20260921/results/e5-300-50-holdout.json) | 87.50% | 93.75% | 50.00% / 68.75% |

| 검증 질문 | baseline 완전/조건 | 선택 후보 완전/조건 |
| --- | ---: | ---: |
| H1 | 1 / 100.00% | 1 / 100.00% |
| H2 | 0 / 75.00% | 1 / 100.00% |
| H3 | 0 / 66.67% | 1 / 100.00% |
| H4 | 1 / 100.00% | 1 / 100.00% |
| H5 | 0 / 50.00% | 0 / 50.00% |
| H6 | 1 / 100.00% | 1 / 100.00% |
| H7 | 0 / 33.33% | 1 / 100.00% |
| H8 | 0 / 50.00% | 1 / 100.00% |

영어 holdout의 완전근거 평균·조건 평균과 필수질문별 완전회수가 baseline보다 떨어지면 baseline을 유지한다. 판정 결과 `holdout_passed=True`, 유지 설정은 `e5-300-50`다. 한국어 점수는 보조로 보고하며 이 선택을 뒤집는 기준으로 사용하지 않았다. 인접페이지 보강과 M3 긴 입력의 holdout 재선택 실험은 수행하지 않았다.

## [EXP-05] 기업 RFP·계약 corpus 미확보

공식 IT 제안요청/계약서 최대3문서60페이지와 독립16질문(선택8/검증8)은 기술논문 corpus와 별도로 계획했다. 실제 원본 다운로드는403/404로 실패했고 스캔형 후보도 평가 가능한 추출문을 확보하지 못했다. 따라서 기업문서 질문·점수·가상의 결과를 만들지 않았다. [후보별 공식 출처와 실패기록](source/model-and-enterprise-evidence-20260921.md)을 참조한다. 기술논문 실험으로 기업업무 품질이나 기업문서용 최적모델을 주장할 수 없다.

도구는 별도 `--pages`, `--dataset`, `--output`으로 기업 corpus를 평가할 수 있지만, 원본해시와8/8분할을 먼저 고정해야 한다. 기업 결과만으로 기술제품 운영모델을 바꾸는 판정은 허용하지 않는다.

## 재현과 검증

기존 결과를 덮어쓰지 않도록 새 출력 디렉터리를 지정한다. 고정 revision snapshot이 `.cache/huggingface/hub`에 준비되어 있어야 한다. 네트워크 모델 다운로드나 유료 LLM 호출은 이 도구에 없다.

```bash
.venv/bin/python scripts/retrieval_experiments.py freeze --cache .cache/huggingface/hub --output outputs/retrieval-reproduction
.venv/bin/python scripts/retrieval_experiments.py prepare --cache .cache/huggingface/hub --output outputs/retrieval-reproduction
.venv/bin/python scripts/retrieval_experiments.py run-all --cache .cache/huggingface/hub --output outputs/retrieval-reproduction
.venv/bin/python scripts/retrieval_experiments.py select --cache .cache/huggingface/hub --output outputs/retrieval-reproduction
.venv/bin/python scripts/retrieval_experiments.py validate --cache .cache/huggingface/hub --output outputs/retrieval-reproduction
.venv/bin/python -m pytest -q tests/test_retrieval_experiments.py
```

실제 출력 디렉터리는 `outputs/retrieval-experiments-20260921`다. [선택 측정 로그](../experiments/retrieval-experiments-20260921/run-all.log)와 [마지막 검증 로그](../experiments/retrieval-experiments-20260921/validate.log), 개별 JSON의 config·hash·raw_ranking·문맥 청크ID·토큰수로 재검산할 수 있다. [원시 문맥 재계산 검사](../experiments/retrieval-experiments-20260921/selection-evidence-check.json)는 선택15설정·480언어질의·1440문맥 점수가 고정 원문 구간에서 다시 계산한 값과 일치함을 기록한다. 프로토콜을 고정한 뒤 코드/질문/모델을 바꾸면 실행이 거부된다. 완료된 결과는 덮어쓰지 않는다. 중단 재개는 같은 hash의 기존 결과와 벡터만 재사용한다.

고정 SHA256:

| 대상 | SHA256 |
| --- | --- |
| 실행 스크립트 | `d2e2c49317d9c300f9a8133029fec0e62dcae3d8c2b52227e00bea1a6d1070e1` |
| 원문 pages | `dffa79e8eed1793f16c217c624af55d6d2f93c629f41d815643082657275fca3` |
| 질문 dataset | `ccc50193422eb087fcbc0474ea107c2086df8ab63054b02c4c590216a6d364e8` |
| 모델 specs | `ceb3bc71ae5c849ec7c1a1921af0ed70c04ffc6a124517cf2c1a765b94b454e3` |
| protocol | `20c5b97000989993f6f8087ae217e73b7c155efb9d6853150c3f7000a868474e` |
| selection | `3db63ce28ec6261068b8c010231bb973cf2f5cff2a59c51aac17ff0ed14182d5` |
| decision | `df610405a0e9852efc8dc36b596e7fc28c8e2b24ecef9f4565c6fa3e2d8467c8` |

작은 두 논문·24독립질문이고 정답 의미의 사람검수가 끝나지 않아 통계적 일반화나 기업 도메인 검증을 주장하지 않는다. 16개 단위테스트는 truncation·0 overlap·복수 span 합집합·문맥예산·holdout 누출 방지·필수질문 비하락·검증 fallback·운영 baseline 동일성·한국어 보조 분리를 확인한다. 원문 인용의 기계적 offset/hash 일치와 사람의 의미검수는 별개다.
