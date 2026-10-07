# SUMMARY

KIVI와 InfiniGen은 서로 다른 KV cache 병목을 줄이는 기술이다. KIVI는 키와 값의 특성에 맞춘 저비트 양자화로 GPU 메모리 사용을 줄인다. InfiniGen은 CPU에 둔 KV cache에서 필요한 토큰을 예측해 GPU로 가져오며 전송량을 줄인다. 두 논문의 모델·장비·워크로드가 다르므로 성능 수치만으로 우열을 정할 수 없다. [1, 물리 p.2·7–8] [2, 물리 p.6·9–13]

기업의 IT 사업 문서 검토에서는 처리량뿐 아니라 인용 정확도, 조항 누락, 변경 추적과 보안·운영 조건이 중요하다. KIVI는 GPU 메모리 용량을, InfiniGen은 CPU-GPU 전송량을 중심으로 적용 가능성을 검토할 수 있다. 두 기술의 업무 적합성은 대표 문서와 동일한 품질 기준을 사용한 후속 실험으로 판단한다.

## 평가 대상과 범위

**소프트웨어 관점:** KIVI의 비대칭 KV cache 양자화.

**하드웨어·시스템 관점:** InfiniGen의 CPU-GPU 메모리 계층과 선택적 전송.

**적용 도메인:** 공개 SK AX AiPMO 설명의 사업 자료 분석 업무를 참고해 RFP·계약서·사업계획서·발주 문서 검토를 설정했다. 두 기술의 적용과 장문·반복·동시 요청은 팀이 정한 실험 가정이다. [5, snapshot block 15]

**판정 방식:** 논문 저자의 실험 결과와 공식 저장소·서비스 설명을 바탕으로 기술 성숙도·시장성·이해관계자·도메인 적용을 평가했다. 업무 효과와 이해관계자의 부담은 기술 구조에서 도출한 팀의 해석이다. 논문 실험의 독립 재현과 기업 업무 적용 시험은 후속 검증 범위다.

<!-- pagebreak -->

# 기술 성숙도

## KIVI: 양자화 설정과 품질의 관계

KIVI는 key cache를 per-channel, value cache를 per-token으로 양자화한다. 그룹으로 묶인 캐시를 양자화하고 잔여 부분은 full precision으로 유지한다. 구현은 CUDA의 역양자화·행렬곱 결합과 Triton의 group-wise 양자화 커널을 사용한다. [1, 물리 p.2·6]

**논문 보고:** Table 3은 Llama-2-7B·13B, Falcon-7B, Mistral-7B의 결과를 비교한다. CoQA·GSM8K는 exact match accuracy, TruthfulQA는 BLEU를 사용했다. Falcon-7B의 MQA 구조에서는 2비트 품질 저하가 커 4비트를 함께 검토해야 한다. 민감도 분석에서 group size를 바꿀 때 residual length는 128로 고정했고, residual length를 바꿀 때 group size는 32로 고정했다. group size 128에서는 성능 저하가 관찰됐다. [1, 물리 p.6–7·9, Table 3·5]

효율 실험은 Llama-2-7B, 단일 NVIDIA A100 80GB, 2비트 KIVI와 FP16 비교, ShareGPT 기반 합성 서비스 워크로드에서 수행했다. 평균 입력은 161토큰, 평균 출력은 338토큰이다. residual length 32·128에서 메모리 한계까지 배치를 늘렸으며, 저자는 최대 배치 증가와 처리량 개선을 보고했다. [1, 물리 p.7–8, Figure 5]

## InfiniGen: 전송 선택과 메모리의 관계

InfiniGen은 CPU 메모리의 KV cache pool에서 attention에 필요한 key/value를 선택해 GPU로 가져온다. 이전 층의 입력을 이용한 예측, partial weight·key cache, 오프라인 query/key weight 변환을 결합한다. CPU 캐시는 모델이 계산한 key/value tensor를 담는다. [2, 물리 p.6–8]

**논문 보고:** 성능 평가의 장비는 RTX A6000 48GB, Xeon Gold 6136, DDR4-2666 96GB, PCIe 3.0×16이다. Figure 14는 OPT-13B, 입력 1,920·출력 128토큰, batch 20의 prefill과 decoding 지연을 비교한다. 저자는 실제 경과 시간(wall-clock time)을 측정했다고 설명한다. 여기에 쓰인 INT4·H2O도 CPU에서 KV cache를 전송하는 비교 구성이다. [2, 물리 p.9–11]

partial weight ratio 민감도는 OPT-6.7B, WinoGrande, 입력 1,920·출력 128토큰, batch 8에서 측정했다. alpha=4를 고정한 실험에서 ratio 0.3을 넘으면 정확도 차이는 크지 않고 메모리 부담은 늘었다. Figure 18의 OPT-13B·길이 2,048·batch 8 단일 블록 지연 분석은 별도 실험이다. [2, 물리 p.12–13, Figure 17·18]

**팀의 성숙도 판단:** 공개 구현과 논문의 연구 실험을 근거로 두 기술을 실험실 검증 수준인 잠정 TRL 4로 평가한다. 기업 문서 검토에 적용하려면 대표 데이터·운영 환경·지속 운용에서의 검증이 필요하다. [1] [2] [3] [4]

<!-- pagebreak -->

# 시장성

공개 논문과 저장소는 두 기술의 구현과 연구 실험을 확인할 수 있는 자료다. 이번에 검토한 자료에서 기업 IT 사업 문서 검토의 실제 채택·고객 성과·장기 운영 결과는 확인되지 않았다. [1–5]

| 평가 질문 | KIVI | InfiniGen |
| --- | --- | --- |
| 어떤 제약을 검토하는가 | KV cache의 GPU 메모리 사용과 모델별 양자화 품질 | CPU–GPU 캐시 이동량과 CPU 메모리·PCIe 제약 |
| 비교할 대안 | 동일 모델의 FP16, 2비트·4비트 KV 설정. weight-only 양자화와는 압축 대상을 구분 | 동일 offloading 조건의 UVM·FlexGen·H2O·INT4. 논문 Figure 14의 비교 구성을 유지 |
| 통합 부담 | CUDA·Triton 커널, 모델 구조와 group size·residual length 설정, 품질 회귀 확인 | 캐시 풀·선택기·prefetch 제어, weight 변환, CPU·GPU·PCIe 관측 |
| 공개 구현 신호 | 논문과 공식 저장소. README는 MIT License를 명시 | 논문과 artifact evaluation용 공식 저장소 |
| 아직 판단할 수 없는 것 | 기업 문서 품질, 운영 총비용, 장애·복구, 장기 유지보수 | 기업 문서 품질, 자원 조달비, 장애·복구, 장기 유지보수 |

구조와 대안 비교의 근거는 KIVI 물리 p.2·6–8, InfiniGen 물리 p.6·9–13 및 두 README다. 표의 통합 부담은 이 구조에서 도출한 팀의 해석이다. [1] [2] [3] [4]

## 비용과 사용 조건

총소유비용은 목표 업무의 요청량·길이·동시성, 하드웨어 비용, 전력, 품질 검토 인력과 장애 대응 시간을 함께 측정해야 산정할 수 있다. 현재 자료로는 금전적 절감액과 투자 수익을 계산하기 어렵다.

KIVI README는 MIT License를 명시한다. 기업 배포에서는 LICENSE 전문과 모델 가중치·의존성의 사용 조건을 함께 검토해야 한다. InfiniGen도 적용 버전의 라이선스 원문과 의존성 확인이 필요하다. [3, License 절] [4]

<!-- pagebreak -->

# 이해관계자

팀이 예상하는 이해관계자별 효과와 검증 부담은 다음과 같다.

| 이해관계자 | 기대할 수 있는 변화 | 함께 부담할 검증 |
| --- | --- | --- |
| 문서 검토자 | 메모리·전송 병목이 줄면 더 많은 문서나 요청을 처리할 여지가 생김 | 인용 정확도, 조항 누락, 변경 추적이 유지되는지 원문과 비교 |
| AI·인프라 운영자 | KIVI의 GPU 캐시 압축, InfiniGen의 선택적 전송을 자원 제약에 맞춰 실험 | 커널·모델 호환성, CPU·GPU 메모리, PCIe 전송, 오류·지연 관측과 롤백 |
| 구매·보안·관리 담당자 | 자원 구성과 처리량의 선택 범위를 검토 | 실제 비용, 라이선스·의존성, 접근통제, tenant 격리, 삭제·감사·복구 정책 |

## 처리량과 검토 품질

KIVI의 저비트 설정은 모델과 과제에 따라 품질 부담이 다르다. InfiniGen은 어떤 캐시를 가져올지 선택한다. 두 경우 모두 일반 벤치마크의 정확도만으로 계약 조항과 인용의 누락률을 대신할 수 없다. 검토자는 처리량 증가와 별개로 업무별 품질 기준을 확인해야 한다. [1, 물리 p.6–9] [2, 물리 p.10–13]

## 운영과 책임

KIVI에서는 비트폭·group size·residual length와 GPU 커널을, InfiniGen에서는 CPU 캐시 풀·prefetch·PCIe 경로와 weight 변환을 관리해야 한다. 이는 문서 검토자의 업무 속도와 운영자의 장애 대응 부담이 서로 다른 기준으로 평가될 수 있음을 뜻한다. 실제 부담의 크기는 아직 측정하지 않았다. [1, 물리 p.2·6] [2, 물리 p.6–8·13]

운영 검증에서는 캐시 보존 기간, 삭제, 접근권한, tenant 격리, 감사와 장애 복구를 확인해야 한다. 검토한 자료에는 이 항목의 구현·검증 결과가 제시돼 있지 않다.

<!-- pagebreak -->

# 도메인 적용

## 적용 시나리오

RFP·계약서·사업계획서·발주 문서에서 요구사항과 근거 문장을 찾고, 일정·산출물·위험 및 문서 간 변경을 검토하는 Agentic AI를 가정한다. [5]

KIVI는 KV cache 메모리 부담을 줄일 후보이고, InfiniGen은 CPU offloading이 가능한 환경에서 캐시 전송을 줄일 후보다. 반복·동시 요청을 얼마나 더 처리할 수 있는지와 업무 품질 유지 여부는 각각의 실험으로 확인해야 한다. [1, 물리 p.2·7–8] [2, 물리 p.6·11–13]

## 무엇을 고정하고 무엇을 바꿀 것인가

| 구분 | 비교 설계 |
| --- | --- |
| 공통 기준 | 모델·정밀도·문서 집합·입출력 길이·동시 요청·장비·소프트웨어 버전을 기록하고 비교 조건을 고정 |
| KIVI | FP16·2비트·4비트를 비교. group size와 residual length를 한 번에 하나씩 변경 |
| InfiniGen | alpha와 partial weight ratio를 한 번에 하나씩 변경. CPU 메모리 용량·PCIe 조건을 기록 |
| 문서 품질 | 조항·인용 정확도, 누락률, 환각률, 근거 추적 성공률, 변경 추적 결과를 정답셋과 대조 |
| 자원·운영 | p50·p95 지연, 처리량, CPU·GPU 메모리, 전송량, 오류·복구 결과를 같은 요청 집합에서 측정 |

이 표는 팀이 제안하는 후속 검증 설계다. 기업 문서의 대표 길이 분포, 반복 질의와 동시 요청 조건을 확보한 뒤 목표 수치를 정한다.

## 적용 판단의 경계

도입 판단에는 생성 과제의 정확도와 처리량 외에 기업 문서의 조항·인용 보존 품질이 필요하다. 대표 문서에서 품질 기준을 충족한 뒤 자원 효율과 운영 조건을 함께 평가한다. 공개 자료에서 확인되지 않은 보안·운영 기능은 후속 시험 항목으로 둔다.

<!-- pagebreak -->

# 관점 간 상충과 한계

## 서로 다른 비용을 함께 본다

KIVI는 저비트 캐시와 설정별 품질·GPU 커널 통합의 관계를, InfiniGen은 선택적 전송과 CPU 메모리·예측 제어의 관계를 갖는다. 사용자는 처리 여지를, 운영자는 관측과 복구 가능성을, 구매·보안 담당자는 비용과 사용 조건을 중시할 수 있다. [1, 물리 p.2·6–8] [2, 물리 p.6–8·13]

## 실험별 비교 조건

| 실험 | 주요 조건 | 해석 가능한 범위 |
| --- | --- | --- |
| KIVI 효율 | Llama-2-7B, A100 80GB, ShareGPT 기반, 평균 입력 161·출력 338, residual 32·128 | 해당 서비스 워크로드의 메모리·처리량 |
| InfiniGen Figure 14 | OPT-13B, RTX A6000, 입력 1,920·출력 128, batch 20 | 해당 offloading 비교 구성의 지연 |
| InfiniGen ratio 민감도 | OPT-6.7B, WinoGrande, 입력 1,920·출력 128, batch 8, alpha 4 | partial weight ratio와 정확도·메모리의 관계 |

InfiniGen Figure 18은 OPT-13B·길이 2,048·batch 8의 단일 블록 지연을 분석한다. 위 ratio 민감도와 모델·분석 단위가 다른 실험이다. [1, 물리 p.7–8] [2, 물리 p.11–13]

## 후속 검증 항목

검토 자료에서는 두 기술을 동일한 모델·장비·문서 집합으로 직접 비교한 기업 업무 결과를 확인하지 못했다. 문서 검토의 인용·조항 연결·변경 추적 품질, 접근통제와 캐시 삭제, 장기간 장애·복구, 실제 비용과 유지보수 부담도 별도 검증이 필요하다.

KIVI는 메모리 제약과 양자화 품질을, InfiniGen은 캐시 전송량과 운영 복잡도를 중심으로 검증할 가치가 있다. 다음 판단의 기준은 목표 문서 업무에서 측정한 품질·지연·자원 사용·운영 부담이다.

<!-- pagebreak -->

# REFERENCE

[1] Zirui Liu et al. (2024). **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**. arXiv 2402.02750v2. [원문 PDF](https://arxiv.org/pdf/2402.02750v2).

[2] Wonbeom Lee et al. (2024). **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**. arXiv 2406.19707v1. [원문 PDF](https://arxiv.org/pdf/2406.19707v1).

[3] KIVI authors. **KIVI official repository README**. [공식 저장소 원문](https://raw.githubusercontent.com/jy-yuan/KIVI/main/README.md). 수집 2026-09-21. 구현과 License 절 참고.

[4] SNU Computer Architecture Lab. **InfiniGen official repository README**. [공식 저장소 원문](https://raw.githubusercontent.com/snu-comparch/InfiniGen/main/README.md). 수집 2026-09-21. artifact evaluation용 구현 설명 참고.

[5] SK AX. **SK AX AiPMO**. [공식 서비스 설명](https://www.skax.co.kr/ax-services/aipmo). 수집 2026-09-21. 사업 자료 분석 업무 설명 참고.

본문의 페이지는 PDF의 물리 페이지다. 논문 확인일은 2026-10-07이며, 웹 자료는 위 수집일의 저장본을 사용했다. 버전·파일 식별 정보와 인용 대조 기록은 [출처 검토 기록](https://github.com/pbjuni1007-cmyk/skala-rag-project/blob/work/agent-integration/docs/reference-check-main08.md)에 있다.
