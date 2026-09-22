# SUMMARY

**KIVI · InfiniGen** · [팀 추론] KIVI는 GPU 메모리·배치 여력을 사용자와 인프라 담당자에게 제공하지만 양자화 회귀와 결과 확인 부담을 남긴다. InfiniGen은 KV 전송 오버헤드 절감을 기대하게 하지만 CPU KV pool, alpha·partial weight, 예측 누락 여부 검증이 운영·사용자 부담이 된다. [5, 물리 p.8] [5, 물리 p.6] [3, 물리 p.6] [3, 물리 p.2] [3, 물리 p.9] [3, 물리 p.13] [1, snapshot block 17, character 0]

**KIVI · InfiniGen** · [팀 추론] KIVI는 GPU KV 메모리가 병목이고 품질 회귀시험을 통과할 때 도입 후보이며, InfiniGen은 장문·대배치의 CPU–GPU KV 전송이 병목이고 CPU 메모리·PCIe를 확보할 때 후보가 된다. 문서 Agent 품질과 운영지표 검증 전에는 우위를 확정하지 않는다. [5, 물리 p.2] [2, snapshot block 1, character 0] [3, 물리 p.2] [3, 물리 p.9] [3, 물리 p.11] [5, 물리 p.6] [3, 물리 p.12] [1, snapshot block 15, character 0]

**InfiniGen** · [팀 추론] InfiniGen: CPU 메모리와 PCIe를 활용하는 장문·대배치 offloading이 병목이면 H2O·FlexGen·INT4 대비 검토 가치가 있지만, GPU 내 KV 양자화만 필요한 환경에는 복잡도가 커질 수 있다. [3, 물리 p.11] [3, 물리 p.2]

**대상 기술:** KIVI / InfiniGen  
**단일 도메인:** 기업의 IT 사업 문서 검토를 지원하는 Agentic AI

**적용 가정:** 공개 AiPMO 사례를 참고한 RFP·계약·사업 문서 검토. 장문과 반복 요청은 팀의 적용 가정이며 SK AX 내부 구조·기술 도입 사실이 아님.

# 기술 성숙도

## KIVI · research_kivi-1

**출처 사실 · research_kivi-1**

KIVI는 키 캐시를 채널별, 값 캐시를 토큰별로 2비트 양자화하고, 스트리밍에 맞지 않는 키 캐시는 토큰 그룹과 잔여 FP 캐시로 분할해 정확도와 처리 효율을 함께 확보한다. [5, 물리 p.2] [5, 물리 p.6]

**조건:** 논문 KIVI v2의 Llama/Llama-2, Falcon, Mistral 계열 평가와 자동회귀 추론의 prefill·decoding 구조를 전제로 한다. 키 캐시는 채널별, 값 캐시는 토큰별로 처리하고, 완전한 그룹을 이루지 못한 잔여 캐시는 FP로 유지한다.

**한계:** 이 원리는 KV 캐시 메모리와 디코딩 비용을 줄이는 방법으로 확인되지만, 기업 IT 문서 검토 Agent의 정확도나 검색·도구 호출 품질까지 직접 검증한 결과는 아니다.

## KIVI · research_kivi-2

**저자 보고 결과 · research_kivi-2**

KIVI는 모델 구조와 설정에 따라 정확도 손실이 커질 수 있다. 특히 이미 KV가 압축된 Falcon의 2비트 설정과 큰 group size는 대표 태스크의 품질 저하 위험을 높인다. [5, 물리 p.6] [5, 물리 p.7] [5, 물리 p.9]

**조건:** KIVI v2 Table 3의 모델·정밀도·비교 기준은 Falcon-7B의 16bit, KIVI-2, KIVI-4이며, 정확도 지표와 세부 데이터셋은 발췌상 일부 미확인이다. Table 5는 Llama2-13B, GSM8K, group size 32·64·128 및 residual length 실험이며, group size 128에서 17.29로 감소했다.

**한계:** 검색 발췌에서 확인된 한계는 Falcon·GSM8K 등 논문 벤치마크에 관한 것이다. 기업 문서 검토의 사실성, 인용 정확도, 장문 반복 요청에서의 손실 크기는 미확인이다.

## KIVI · research_kivi-3

**저자 보고 결과 · research_kivi-3**

KIVI의 효율 이득은 ShareGPT 기반 합성 서비스 워크로드에서 확인됐다. Llama-2-7B와 A100 80GB 환경에서 FP16 대비 유사 메모리로 배치와 처리량을 비교한 결과다. [5, 물리 p.7] [5, 물리 p.8]

**조건:** KIVI v2 Figure 5의 효율 실험은 ShareGPT 기반 합성 워크로드, 평균 입력 161토큰·출력 338토큰, Llama-2-7B, 2비트 KIVI residual length 32·128, FP16 baseline, 단일 NVIDIA A100 80GB를 사용했다. 배치는 메모리 한계까지 증가시켜 비교했으며 시작·최종 배치 수, 측정 소프트웨어와 반복 횟수는 발췌상 미확인이다. 지표는 최대 메모리와 throughput이다.

**한계:** 이는 실제 기업 Agent 서비스가 아닌 합성 워크로드의 단일 GPU 실험이다. 문서 길이·동시 요청·출력 분포가 달라지면 동일한 처리량과 배치 이득을 보장할 수 없다.

## KIVI · research_kivi-4

**팀 추론 · research_kivi-4**

KIVI의 공개 근거 기반 잠정 TRL은 4로 판단한다. 공개 구현과 여러 LLM·벤치마크의 실험실 검증은 있으나, 대표적인 기업 IT 문서 Agent 운용환경과 지속 운영까지의 실증은 확인되지 않는다. [5, 물리 p.6] [5, 물리 p.1] [5, 물리 p.13]

**조건:** 판단 근거는 KIVI 논문 v2와 공개 저장소 정보다. 논문은 Hugging Face Transformers 기반 구현, CUDA 역양자화·행렬곱 융합, Triton 커널, Llama/Llama-2·Falcon·Mistral 평가, LM-Eval·LongBench·Needle-in-a-Haystack 실험을 제시한다. 실험실 GPU 조건과 공개 코드까지는 확인되지만 실제 기업 문서 검토 Agent의 관련환경·운용환경 시스템 시연 조건은 미확인이다.

**한계:** TRL은 논문이 인증한 값이 아니라 공개 실험 범위에 대한 팀의 잠정 해석이다. 코드 공개와 다수 벤치마크만으로 대표 사용조건 검증인 TRL 5 이상으로 높이지 않았으며, 라이선스·운영 SLA·다중 GPU 및 Agent 도구 연동 검증은 확인되지 않았다.

## InfiniGen · research_infinigen-1

**출처 사실 · research_infinigen-1**

InfiniGen은 다음 레이어의 attention pattern을 미리 추정해 필요한 KV만 GPU로 전송하는 구조로, 오프라인 weight skewing과 CPU KV pool 관리가 전송량 절감의 핵심이다. [3, 물리 p.2] [3, 물리 p.9]

**조건:** 근거 문서는 arXiv:2406.19707v1(2024-06-28)이다. 디코딩 중 Layer i−1의 attention input, Layer i의 partial query weight와 partial key cache로 다음 attention pattern을 추정하고, alpha 임계값으로 KV를 동적으로 선택한다. prefill 단계에서 partial weight를 생성하며, CPU 메모리의 KV pool에서 비빈번 토큰을 제거한다.

**한계:** 이는 offloading 기반 생성 추론에서의 KV 전송·관리 메커니즘에 대한 확인이며, 기업 IT 사업 문서 검토 Agentic AI의 실제 워크플로우에서 효과가 검증된 것은 아니다.

## InfiniGen · research_infinigen-2

**저자 보고 결과 · research_infinigen-2**

InfiniGen은 attention pattern의 반복 간 변화와 레이어·query별 KV 요구량 차이를 처리하지만, alpha와 partial weight ratio가 정확도·지연·메모리 간 절충을 만들며 오류율은 확인되지 않았다. [3, 물리 p.5] [3, 물리 p.6] [3, 물리 p.13]

**조건:** 논문은 partial weight ratio를 높여도 일정 범위 이후 accuracy 차이가 크지 않다고 보고했으며 ratio 0.3을 선택했다. ratio가 두 배가 되면 partial weights와 key cache의 메모리 overhead가 두 배가 된다고 설명한다. alpha와 ratio 민감도는 OPT-6.7B, 입력 1920, 출력 128, batch 8, WinoGrande 조건에서 평가했다.

**한계:** 제공된 발췌에는 attention 예측 오류율, PCIe 세대별 민감도, 장시간 반복 요청에서의 안정성 수치가 없다. ratio 0.3 선택은 논문 실험의 accuracy와 memory overhead 절충이며 보편적 최적값으로 볼 수 없다.

## InfiniGen · research_infinigen-3

**저자 보고 결과 · research_infinigen-3**

InfiniGen은 RTX A6000·PCIe 3.0×16 환경에서 OPT-13B의 2048 길이·batch 20 지연을 FlexGen·H2O·INT4와 비교해 1.63×–32.93× speedup으로 보고했다. [3, 물리 p.9] [3, 물리 p.11]

**조건:** 출처 버전은 arXiv:2406.19707v1(2024-06-28)이다. Figure 14의 지연 실험은 OPT-13B, sequence length 2048(입력 1920·출력 128), batch 20, NVIDIA RTX A6000 48GB, Intel Xeon Gold 6136, DDR4-2666 96GB, PCIe 3.0×16에서 수행했고 prefill·decoding latency를 FlexGen, UVM, H2O(KV budget 20%), FlexGen+INT4와 비교했다. 지표는 inference latency이며 시뮬레이션이 아닌 시스템 실행 결과로 제시된다. 모델 정밀도와 해당 latency 실험의 데이터셋은 미확인이다. 별도 정확도 실험은 OPT 6.7B/13B/30B 및 Llama-2 7B/13B, 5-shot COPA·OpenBookQA·WinoGrande·PIQA·RTE와 WikiText-2·PTB를 사용했다.

**한계:** 수치는 논문이 제시한 특정 단일 시스템·모델·길이·배치의 실측 결과이며, 다른 GPU·PCIe 세대·동시성 또는 목표 업무에 직접 순위화할 수 없다. 기업 문서 검토 데이터와 Agentic AI 업무 지연은 검증되지 않았다.

## InfiniGen · research_infinigen-4

**팀 추론 · research_infinigen-4**

InfiniGen의 공개 논문 근거상 잠정 TRL은 4 수준(보수적 범위 3~4)이며, 여러 LLM의 실험실 검증은 있으나 기업 업무 운용과 독립 재현까지는 실증되지 않았다. [3, 물리 p.2] [3, 물리 p.9] [3, 물리 p.14]

**조건:** 판단 기준은 arXiv:2406.19707v1(2024-06-28)의 논문 구현과 RTX A6000 기반 실험, OPT·Llama-2 모델 크기 변화, batch·sequence length 변화, downstream accuracy·perplexity·latency·memory 분석이다. 이는 실험실 시스템 검증에 해당하는 근거로 보았으며, 대표 사용조건 또는 실제 지속운용 검증으로 확대하지 않았다. 공개 저장소가 source_metadata에 식별되지만, 제공 발췌만으로 독립 실행 성공이나 라이선스 적합성을 확인하지 않았다.

**한계:** TRL은 논문이 인증한 값이 아니라 공개 실험 범위에 대한 팀의 보수적 해석이다. 논문 구현은 대표 LLM·배치·sequence length 실험을 제공하지만, 기업 IT 사업 문서 검토 Agentic AI의 장문·반복·동시 요청, 운영환경 SLA, 장애복구, 보안, 독립 재현과 공식 라이선스·실행 절차는 제공 발췌에서 확인되지 않는다.

# 시장성

| 비교 질문 | KIVI | InfiniGen |
| --- | --- | --- |
| 채택 동기·공개 신호 | **팀 추론 · market-1**<br><br>KIVI: 공개 논문·공식 코드와 LLM 벤치마크는 생태계 신호지만, 기업 IT 문서 검토 Agent의 공개 채택·상용 운영은 제공 범위에서 확인되지 않는다. [5, 물리 p.1]<br><br>[조건·한계·원문](citation_review.md#market-1) | **팀 추론 · market-4**<br><br>InfiniGen: 공개 논문·저장소와 OPT·Llama-2 실험은 offloading 구현의 성숙도 신호지만, 기업 IT 문서 검토 Agent의 공개 채택·상용 운영은 제공 범위에서 확인되지 않는다. [3, 물리 p.2] [3, 물리 p.9] [4, snapshot block 8, character 0]<br><br>[조건·한계·원문](citation_review.md#market-4) |
| 대안·연동 | **팀 추론 · market-2**<br><br>KIVI: GPU KV 메모리와 배치 용량이 핵심 병목이면 AWQ·GPTQ나 시스템형 vLLM·S3보다 직접적인 선택지지만, MQA·GQA 모델은 4비트와 품질 회귀를 우선 검토해야 한다. [5, 물리 p.8] [5, 물리 p.9] [2, snapshot block 1, character 0]<br><br>[조건·한계·원문](citation_review.md#market-2) | **팀 추론 · market-5**<br><br>InfiniGen: CPU 메모리와 PCIe를 활용하는 장문·대배치 offloading이 병목이면 H2O·FlexGen·INT4 대비 검토 가치가 있지만, GPU 내 KV 양자화만 필요한 환경에는 복잡도가 커질 수 있다. [3, 물리 p.11] [3, 물리 p.2]<br><br>[조건·한계·원문](citation_review.md#market-5) |
| 비용·유지 부담 | **팀 추론 · market-3**<br><br>KIVI: 금전적 도입·운영비는 산정할 수 없으며, Transformers 통합과 CUDA·Triton 커널 검증, 잔여 FP 캐시 관리 및 업무별 품질 회귀시험이 주요 부담으로 남는다. [5, 물리 p.6] [5, 물리 p.2]<br><br>[조건·한계·원문](citation_review.md#market-3) | **팀 추론 · market-6**<br><br>InfiniGen: 금전적 도입·운영비는 공개 자료로 산정할 수 없고, CPU KV pool·PCIe 전송 관리와 partial weight·alpha 설정의 메모리·품질 검증 부담을 추가한다. [3, 물리 p.9] [3, 물리 p.13] [3, 물리 p.12]<br><br>[조건·한계·원문](citation_review.md#market-6) |

# 이해관계자

| 비교 질문 | KIVI | InfiniGen |
| --- | --- | --- |
| 문서 검토자 | **팀 추론 · stakeholder-1**<br><br>KIVI는 KV 캐시 양자화로 문서 검토 Agent의 동시 처리 여력을 높일 가능성이 있지만, RFP·계약서의 인용·요약 정확도는 별도 검증해야 해 사용자 확인 부담이 남는다. [5, 물리 p.8] [1, snapshot block 15, character 0]<br><br>[조건·한계·원문](citation_review.md#stakeholder-1) | **적용 가정 · stakeholder-4**<br><br>InfiniGen은 CPU KV pool에서 필요한 토큰만 GPU로 가져와 장문·반복 요청의 지연 완화를 기대하게 하지만, 예측 누락이 문서 인용·근거 품질에 미치는 영향은 사용자가 검증해야 한다. [3, 물리 p.2] [3, 물리 p.9] [1, snapshot block 15, character 0]<br><br>[조건·한계·원문](citation_review.md#stakeholder-4) |
| AI·인프라 운영자 | **팀 추론 · stakeholder-2**<br><br>KIVI는 Hugging Face·CUDA·Triton 기반 구현으로 운영 통합을 검토할 수 있으나, group size·residual length별 품질과 메모리·지연 회귀를 관측하는 부담은 운영자에게 남는다. [5, 물리 p.6] [5, 물리 p.7]<br><br>[조건·한계·원문](citation_review.md#stakeholder-2) | **팀 추론 · stakeholder-5**<br><br>InfiniGen은 CPU 메모리와 GPU 사이의 동적 prefetch로 전송 병목을 줄일 수 있지만, alpha·partial weight ratio·PCIe 상태와 CPU·GPU 자원을 함께 관측하는 운영 부담이 커진다. [3, 물리 p.13] [3, 물리 p.9] [3, 물리 p.12]<br><br>[조건·한계·원문](citation_review.md#stakeholder-5) |
| 구매·보안·관리 담당자 | **팀 추론 · stakeholder-3**<br><br>KIVI는 GPU 메모리 절감과 공개 구현으로 인프라 부담을 낮출 여지가 있지만, 모델 구조별 2비트 손실 위험과 라이선스·보안·결과 책임 기준 확인이 구매 승인 조건이다. [5, 물리 p.6] [5, 물리 p.1] [5, 물리 p.9]<br><br>[조건·한계·원문](citation_review.md#stakeholder-3) | **팀 추론 · stakeholder-6**<br><br>InfiniGen은 GPU 증설 대신 CPU 메모리·PCIe 구성을 활용하는 선택지지만, 특정 하드웨어 의존성과 CPU 내 KV 보관의 보안·라이선스·복구 기준 확인이 구매 승인의 전제다. [3, 물리 p.9] [3, 물리 p.11] [3, 물리 p.2]<br><br>[조건·한계·원문](citation_review.md#stakeholder-6) |

# 도메인 적용

| 비교 질문 | KIVI | InfiniGen |
| --- | --- | --- |
| 적합 조건 | **팀 추론 · domain-1**<br><br>KIVI는 KV 캐시를 2비트로 줄이고 잔여 구간은 FP로 유지하므로, 장문·반복 문서 검토 Agent의 동시 처리 후보가 되지만 업무 품질 적합성은 간접 근거다. [5, 물리 p.2] [5, 물리 p.6] [5, 물리 p.8] [1, snapshot block 15, character 0]<br><br>[조건·한계·원문](citation_review.md#domain-1) | **적용 가정 · domain-4**<br><br>InfiniGen은 CPU KV pool에 캐시를 두고 다음 레이어에 필요한 항목만 GPU로 prefetch하므로 장문·반복 요청에 맞을 가능성이 있지만, CPU 메모리와 PCIe 계층을 전제로 한다. [3, 물리 p.2] [3, 물리 p.6] [1, snapshot block 15, character 0]<br><br>[조건·한계·원문](citation_review.md#domain-4) |
| 정확도·운영 위험 | **저자 보고 결과 · domain-2**<br><br>KIVI는 2비트 압축 이득이 모델 구조와 설정에 좌우된다. Falcon의 이미 압축된 KV에서는 4비트가 필요할 수 있고, 큰 group size는 대표 태스크 정확도를 낮출 위험이 있다. [5, 물리 p.6] [5, 물리 p.7] [5, 물리 p.9]<br><br>[조건·한계·원문](citation_review.md#domain-2) | **팀 추론 · domain-5**<br><br>논문 결과: InfiniGen은 필요한 KV를 예측해 전송량을 줄이며, partial weight ratio를 높이면 partial weights와 key cache의 메모리 사용량이 증가한다. 팀 추론: 예측 누락이 기업 문서의 근거 보존과 응답 지연에 미치는 영향은 업무 데이터로 확인해야 한다. [3, 물리 p.5] [3, 물리 p.9] [3, 물리 p.13]<br><br>[조건·한계·원문](citation_review.md#domain-5) |
| 확인할 실험 | **팀 추론 · domain-3**<br><br>KIVI의 논문 평가는 일반·장문 생성과 메모리·처리량을 다루지만, 기업 문서 Agent의 정확도와 p95 지연을 검증하지 않아 업무 도입 판정에는 별도 시험이 필요하다. [5, 물리 p.6] [5, 물리 p.13] [5, 물리 p.8] [1, snapshot block 17, character 0]<br><br>[조건·한계·원문](citation_review.md#domain-3) | **팀 추론 · domain-6**<br><br>InfiniGen은 실제 GPU·CPU·PCIe 시스템에서 장문 batch 추론 지연을 측정했지만, 그 speedup은 기업 문서 Agent의 품질·동시성 성과로 전환되지 않아 별도 검증이 필요하다. [3, 물리 p.9] [3, 물리 p.11] [3, 물리 p.12]<br><br>[조건·한계·원문](citation_review.md#domain-6) |

# 관점 간 상충과 한계

## both · synthesis-1

**팀 추론 · synthesis-1**

KIVI는 GPU 메모리·배치 여력을 사용자와 인프라 담당자에게 제공하지만 양자화 회귀와 결과 확인 부담을 남긴다. InfiniGen은 KV 전송 오버헤드 절감을 기대하게 하지만 CPU KV pool, alpha·partial weight, 예측 누락 여부 검증이 운영·사용자 부담이 된다. [5, 물리 p.8] [5, 물리 p.6] [3, 물리 p.6] [3, 물리 p.2] [3, 물리 p.9] [3, 물리 p.13] [1, snapshot block 17, character 0]

**조건:** KIVI 2402.02750v2 Figure 5는 ShareGPT 기반 합성 워크로드, Llama-2-7B, 2비트 KIVI와 FP16(16비트) baseline, 평균 입력 161·출력 338토큰, 단일 NVIDIA A100 80GB에서 배치를 메모리 한계까지 늘려 최대 메모리와 throughput을 비교했다. 보고값은 최대 4배 batch와 2.35배∼3.47배 throughput이며 초기·최종 batch, 측정 소프트웨어·반복 횟수와 시뮬레이션 여부는 제공 발췌에서 미확인이다. KIVI 품질 부담은 Falcon-7B Table 3의 16비트·KIVI-2·KIVI-4 비교를 사용했으며 세부 데이터셋·지표는 일부 미확인이다. InfiniGen은 2406.19707v1의 offloading, alpha 선택, OPT-6.7B·입력 1920·출력 128·batch 8·WinoGrande 민감도와 partial weight ratio 0.3을 기준으로 했다. 업무 역할은 반복·정형 검토를 AI가 하고 전문가가 최종 판단하는 조건이다.

**한계:** 역할별 효익과 부담은 논문 실험과 업무 설명을 연결한 팀 해석이다. 기업 문서의 조항·인용·요약 품질, Agent 도구 호출, attention 예측 오류율, 장기 반복 운영, SLA·보안·공식 라이선스는 제공 근거에서 확인되지 않으며 논문 벤치마크 손실을 업무 오류로 환산하지 않았다.

## both · synthesis-2

**팀 추론 · synthesis-2**

KIVI는 GPU KV 메모리가 병목이고 품질 회귀시험을 통과할 때 도입 후보이며, InfiniGen은 장문·대배치의 CPU–GPU KV 전송이 병목이고 CPU 메모리·PCIe를 확보할 때 후보가 된다. 문서 Agent 품질과 운영지표 검증 전에는 우위를 확정하지 않는다. [5, 물리 p.2] [2, snapshot block 1, character 0] [3, 물리 p.2] [3, 물리 p.9] [3, 물리 p.11] [5, 물리 p.6] [3, 물리 p.12] [1, snapshot block 15, character 0]

**조건:** KIVI 2402.02750v2의 채널별 key·토큰별 value KV 양자화와 MQA/GQA에서 KIVI-4를 권고한 조건을 GPU KV 메모리 병목 판단에 사용하고, LM-Eval의 CoQA exact match·TruthfulQA BLEU·GSM8K exact match를 품질 게이트로 둔다. InfiniGen 2406.19707v1의 후보 조건은 CPU KV pool과 선택적 prefetch, Figure 14의 OPT-13B·입력 1920·출력 128(시퀀스 2048)·batch 20·RTX A6000 48GB·Xeon Gold 6136·DDR4-2666 96GB·PCIe 3.0×16에서 UVM·H2O(KV budget 20%)·FlexGen·FlexGen+INT4와 prefill/decoding latency를 비교한 시스템 실행이다. 해당 latency 실험의 모델 정밀도·데이터셋은 미확인이고, 선행 accuracy는 WinoGrande로 평가됐다. 목표 업무는 RFP·계약서·사업계획서·발주 문서의 조항·요구사항 추출을 대상으로 인용 precision/recall, 요약 사실성, 도구 호출 성공률, p95 지연, 최대 동시성과 GPU·CPU 메모리·전송량을 full-KV baseline과 별도 측정한다.

**한계:** 시장 도입·업무 적합성은 공개 채택이나 생산운영으로 확인된 것이 아니다. KIVI와 InfiniGen의 선행 실험은 모델·정밀도·입출력 길이·배치·GPU·메모리 계층이 달라 직접 우열을 비교할 수 없고, 문서 Agent의 보안·라이선스·SLA도 제공 근거에서 미확인이다.

## 남은 근거 공백

### 업무 품질과 서비스 성능

기업 IT 문서의 요구사항·조항 추출, 요약, 인용, 질의응답과 도구 호출을 대상으로 두 기술을 검증한 자료가 부족하다. 실제 문서 길이·동시 요청 조건에서 정확도, 지연, 비용과 안정성을 함께 측정해야 한다.

관련 검토: [research_kivi-1](gap_review.md#gap-research_kivi-1--research_kivi), [research_kivi-2](gap_review.md#gap-research_kivi-2--research_kivi), [research_infinigen-1](gap_review.md#gap-research_infinigen-1--research_infinigen), [market-1](gap_review.md#gap-market-1--market), [market-2](gap_review.md#gap-market-2--market), [stakeholder-1](gap_review.md#gap-stakeholder-1--stakeholder), [stakeholder-2](gap_review.md#gap-stakeholder-2--stakeholder), [domain-1](gap_review.md#gap-domain-1--domain).

### 동일 조건 비교

모델·정밀도·입출력 길이·배치·GPU·CPU 메모리·PCIe를 맞춘 직접 비교가 없어, 서로 다른 논문의 수치로 기술 우열을 결정할 수 없다.

관련 검토: [research_kivi-3](gap_review.md#gap-research_kivi-3--research_kivi), [stakeholder-3](gap_review.md#gap-stakeholder-3--stakeholder), [domain-3](gap_review.md#gap-domain-3--domain).

### 설정과 하드웨어 민감도

InfiniGen의 예측 오류율과 PCIe·CPU 메모리 구성별 민감도, KIVI의 기업 문서 근거 보존 손실을 추가로 확인해야 한다.

관련 검토: [research_infinigen-2](gap_review.md#gap-research_infinigen-2--research_infinigen), [stakeholder-4](gap_review.md#gap-stakeholder-4--stakeholder), [domain-4](gap_review.md#gap-domain-4--domain).

### 도입·운영 조건

공개 상용 채택, 독립 재현 절차, 라이선스 적합성, 유지보수 비용, 보안·접근통제, 장애 복구, SLA와 결과 책임분담을 확인해야 한다. 현재 검색 근거만으로 도입 승인을 판단하기는 어렵다.

관련 검토: [research_infinigen-3](gap_review.md#gap-research_infinigen-3--research_infinigen), [market-3](gap_review.md#gap-market-3--market), [stakeholder-5](gap_review.md#gap-stakeholder-5--stakeholder), [domain-5](gap_review.md#gap-domain-5--domain).

### 적용 시나리오의 범위

SK AX의 장문·반복·동시 요청은 공개 AiPMO 사례를 바탕으로 한 적용 가정이다. 실제 내부 구조·업무량이나 두 기술의 도입 사실을 확인한 것은 아니다.

관련 검토: [domain-2](gap_review.md#gap-domain-2--domain).

공백별 원문과 판단 근거는 [공백 검수표](gap_review.md)에 보존했다.

# REFERENCE

[1] SK AX (발행일 미상). **SK AX AiPMO**. SK AX, 스냅샷 6dbb089c1432. 조회 2026-09-21.  
https://www.skax.co.kr/ax-services/aipmo

[2] KIVI authors (발행일 미상). **More results on LongBench**. GitHub · KIVI, 스냅샷 74a7fdff77c7. 조회 2026-09-21.  
https://raw.githubusercontent.com/jy-yuan/KIVI/main/docs/long_bench.md

[3] Wonbeom Lee et al (2024). **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**. arXiv, 2406.19707v1. 조회 2026-09-22.  
https://arxiv.org/pdf/2406.19707v1

[4] SNU Computer Architecture Lab (발행일 미상). **InfiniGen official repository README**. GitHub · InfiniGen, 스냅샷 f6a08e32c16d. 조회 2026-09-21.  
https://raw.githubusercontent.com/snu-comparch/InfiniGen/main/README.md

[5] Zirui Liu et al (2024). **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**. arXiv, 2402.02750v2. 조회 2026-09-22.  
https://arxiv.org/pdf/2402.02750v2
