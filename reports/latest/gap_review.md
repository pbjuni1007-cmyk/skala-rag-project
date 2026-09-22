# 근거 공백 검수

원래 공백을 삭제하지 않고 종합 시점의 판단과 근거를 보존합니다. 해소 여부의 의미 검수는 사람에게 남깁니다.

## gap-research_kivi-1 · research_kivi

**원래 공백:** SK AX의 장문·반복·동시 요청 조건에서 KIVI의 정확도, 지연시간, 안정성은 직접 검증되지 않았다.

**종합 판단:** unresolved

**판단 이유:** KIVI의 일반 벤치마크와 ShareGPT 합성 워크로드에서 메모리·처리량 결과는 확인되지만, SK AX의 장문·반복·동시 요청 조건에서 정확도·지연시간·안정성을 직접 측정한 근거는 남아 있다.

**근거 주장:** [research_kivi-3](citation_review.md#research_kivi-3), [domain-3](citation_review.md#domain-3), [stakeholder-1](citation_review.md#stakeholder-1)

**사람 검수:** 미검수

## gap-research_kivi-2 · research_kivi

**원래 공백:** 기업 IT 사업 문서 검토 Agent의 질의응답·요약·인용·도구 호출 품질에 대한 KIVI 전용 실험은 확인되지 않았다.

**종합 판단:** unresolved

**판단 이유:** KIVI의 LM-Eval·LongBench·NIAH 및 합성 서비스 실험은 확인되지만, 기업 IT 사업 문서 검토 Agent의 질의응답·요약·인용·도구 호출 품질을 KIVI로 직접 평가한 전용 실험은 제공된 범위에서 확인되지 않는다.

**근거 주장:** [research_kivi-1](citation_review.md#research_kivi-1), [domain-3](citation_review.md#domain-3), [stakeholder-2](citation_review.md#stakeholder-2)

**사람 검수:** 미검수

## gap-research_kivi-3 · research_kivi

**원래 공백:** KIVI와 InfiniGen을 동일 모델·GPU·배치·컨텍스트 길이에서 직접 비교한 근거는 제공된 발췌에 없다.

**종합 판단:** unresolved

**판단 이유:** 기존 주장들은 KIVI와 InfiniGen의 실험 조건과 비교 대상이 서로 다르며 동일 모델·GPU·배치·컨텍스트 길이의 직접 비교가 없음을 확인한다. 따라서 두 기술의 상대 성능을 판정할 공통 조건 근거는 남아 있다.

**근거 주장:** [market-2](citation_review.md#market-2), [market-5](citation_review.md#market-5), [domain-6](citation_review.md#domain-6)

**사람 검수:** 미검수

## gap-research_infinigen-1 · research_infinigen

**원래 공백:** 기업의 IT 사업 문서 검토 Agentic AI에서 장문·반복·동시 요청을 처리할 때의 정확도, 지연, 비용, 운영 안정성은 미검증이다.

**종합 판단:** unresolved

**판단 이유:** InfiniGen의 특정 OPT·Llama-2 및 RTX A6000 시스템 실험에서 지연·정확도·메모리 결과는 확인되지만, 기업 IT 사업 문서 검토 Agent의 장문·반복·동시 요청에 대한 정확도·지연·비용·운영 안정성은 직접 검증되지 않았다.

**근거 주장:** [research_infinigen-3](citation_review.md#research_infinigen-3), [research_infinigen-4](citation_review.md#research_infinigen-4), [market-6](citation_review.md#market-6), [domain-6](citation_review.md#domain-6)

**사람 검수:** 미검수

## gap-research_infinigen-2 · research_infinigen

**원래 공백:** InfiniGen의 attention prediction 오류율과 PCIe 세대·CPU 메모리 구성별 민감도는 제공 발췌에서 확인되지 않았다.

**종합 판단:** unresolved

**판단 이유:** OPT-6.7B·WinoGrande 조건에서 alpha와 partial weight ratio의 민감도는 일부 확인되지만, attention prediction 오류율과 PCIe 세대·CPU 메모리 구성별 민감도는 제공된 발췌에서 확인되지 않아 공백이 남아 있다.

**근거 주장:** [research_infinigen-2](citation_review.md#research_infinigen-2), [domain-5](citation_review.md#domain-5), [stakeholder-5](citation_review.md#stakeholder-5)

**사람 검수:** 미검수

## gap-research_infinigen-3 · research_infinigen

**원래 공백:** 공식 저장소의 실행 재현 절차와 라이선스 조건은 제공 발췌만으로 확인되지 않았다.

**종합 판단:** unresolved

**판단 이유:** 논문의 RTX A6000 기반 시스템 실행 조건과 공식 저장소 식별은 확인됐지만, 저장소의 단계별 실행 재현 절차와 라이선스 조건은 확인되지 않아 공백이 남는다.

**근거 주장:** [research_infinigen-3](citation_review.md#research_infinigen-3), [research_infinigen-4](citation_review.md#research_infinigen-4), [stakeholder-6](citation_review.md#stakeholder-6)

**사람 검수:** 미검수

## gap-market-1 · market

**원래 공백:** 두 기술 모두 기업 IT 사업 문서 검토 Agent의 사실성, 인용 정확도, 요약·질의응답·도구 호출 품질을 직접 검증한 결과가 제공되지 않았다.

**종합 판단:** unresolved

**판단 이유:** 두 기술의 일반 LLM 벤치마크와 메모리·처리량·시스템 지연 실험은 확인되지만, 기업 IT 사업 문서 검토 Agent의 사실성·인용·요약·질의응답·도구 호출을 직접 검증한 결과는 없다.

**근거 주장:** [domain-3](citation_review.md#domain-3), [stakeholder-1](citation_review.md#stakeholder-1), [stakeholder-4](citation_review.md#stakeholder-4), [domain-6](citation_review.md#domain-6)

**사람 검수:** 미검수

## gap-market-2 · market

**원래 공백:** SK AX의 장문·반복·동시 요청 조건에서 두 기술의 지연시간, 정확도, 장애복구, 보안, 운영비를 검증한 자료가 없다.

**종합 판단:** unresolved

**판단 이유:** 논문은 제한된 모델·하드웨어·길이·배치에서 성능을 측정했지만, SK AX의 장문·반복·동시 요청 가정에 대한 지연·정확도·장애복구·보안·운영비 검증은 확인되지 않았다.

**근거 주장:** [research_kivi-3](citation_review.md#research_kivi-3), [research_infinigen-3](citation_review.md#research_infinigen-3), [market-3](citation_review.md#market-3), [market-6](citation_review.md#market-6), [stakeholder-2](citation_review.md#stakeholder-2), [stakeholder-5](citation_review.md#stakeholder-5), [stakeholder-6](citation_review.md#stakeholder-6)

**사람 검수:** 미검수

## gap-market-3 · market

**원래 공백:** 두 기술의 공식 라이선스 적합성, 유지보수 비용, 독립 재현 절차와 공개 상용 채택 사례는 제공된 출처 범위에서 확인되지 않았다.

**종합 판단:** unresolved

**판단 이유:** 논문과 공식 저장소의 공개 및 실험은 확인되지만, 두 기술의 공식 라이선스 적합성·유지보수 비용·독립 재현 절차·공개 상용 채택을 확인할 근거는 남아 있지 않다.

**근거 주장:** [research_kivi-4](citation_review.md#research_kivi-4), [research_infinigen-4](citation_review.md#research_infinigen-4), [market-1](citation_review.md#market-1), [market-3](citation_review.md#market-3), [market-4](citation_review.md#market-4), [market-6](citation_review.md#market-6), [stakeholder-6](citation_review.md#stakeholder-6)

**사람 검수:** 미검수

## gap-stakeholder-1 · stakeholder

**원래 공백:** 기업 IT 사업 문서 검토에서 두 기술의 사실성·인용·요약·도구 호출 품질을 직접 검증한 실험은 확인되지 않았다.

**종합 판단:** unresolved

**판단 이유:** KIVI의 처리량·메모리 실험과 InfiniGen의 시스템 지연·일반 정확도 실험은 확인되지만, 기업 IT 문서 검토에서 사실성·인용·요약·도구 호출 품질을 직접 검증한 실험은 없다.

**근거 주장:** [stakeholder-1](citation_review.md#stakeholder-1), [stakeholder-4](citation_review.md#stakeholder-4), [domain-3](citation_review.md#domain-3), [domain-6](citation_review.md#domain-6)

**사람 검수:** 미검수

## gap-stakeholder-2 · stakeholder

**원래 공백:** SK AX의 장문·반복·동시 요청 조건에서 두 기술의 정확도·지연·비용·운영 안정성을 직접 비교한 근거가 없다.

**종합 판단:** unresolved

**판단 이유:** KIVI는 ShareGPT·A100 기반 처리량·메모리 결과, InfiniGen은 RTX A6000·PCIe 3.0 기반 지연 결과를 각각 제시했지만, SK AX의 장문·반복·동시 요청에서 정확도·지연·비용·운영 안정성을 동일 조건으로 직접 비교한 근거는 남아 있다.

**근거 주장:** [research_kivi-3](citation_review.md#research_kivi-3), [research_infinigen-3](citation_review.md#research_infinigen-3), [domain-3](citation_review.md#domain-3), [domain-6](citation_review.md#domain-6), [market-3](citation_review.md#market-3), [market-6](citation_review.md#market-6)

**사람 검수:** 미검수

## gap-stakeholder-3 · stakeholder

**원래 공백:** KIVI와 InfiniGen을 동일 모델·GPU·배치·컨텍스트 길이에서 비교한 근거가 없다.

**종합 판단:** unresolved

**판단 이유:** 두 기술의 실험은 모델, GPU, 배치와 입력·출력 길이 및 비교 기준이 서로 달라 동일 모델·GPU·배치·컨텍스트 길이의 직접 비교를 해소하지 못한다.

**근거 주장:** [research_kivi-3](citation_review.md#research_kivi-3), [research_infinigen-3](citation_review.md#research_infinigen-3), [market-2](citation_review.md#market-2), [market-5](citation_review.md#market-5), [domain-6](citation_review.md#domain-6)

**사람 검수:** 미검수

## gap-stakeholder-4 · stakeholder

**원래 공백:** InfiniGen의 attention prediction 오류율과 PCIe 세대·CPU 메모리 구성별 민감도는 제공 발췌에서 확인되지 않았다.

**종합 판단:** unresolved

**판단 이유:** InfiniGen의 alpha·partial weight ratio 절충과 단일 PCIe 3.0 시스템 조건은 확인되지만, attention prediction 오류율과 PCIe 세대·CPU 메모리 구성별 민감도 수치는 제공 근거에서 확인되지 않는다.

**근거 주장:** [research_infinigen-2](citation_review.md#research_infinigen-2), [research_infinigen-3](citation_review.md#research_infinigen-3), [stakeholder-5](citation_review.md#stakeholder-5), [domain-5](citation_review.md#domain-5), [market-6](citation_review.md#market-6)

**사람 검수:** 미검수

## gap-stakeholder-5 · stakeholder

**원래 공백:** 두 기술의 기업 도입을 위한 공식 라이선스, 보안 통제, 장애복구, 운영 SLA와 문서 검토 결과의 책임분담은 추가 확인이 필요하다.

**종합 판단:** unresolved

**판단 이유:** 논문과 공개 저장소는 구현·실험 범위와 일부 통합 부담을 보여주지만, 두 기술의 공식 라이선스 적합성, 보안 통제, 장애복구, 운영 SLA 및 문서 검토 결과의 책임분담은 확인되지 않았다.

**근거 주장:** [research_kivi-4](citation_review.md#research_kivi-4), [research_infinigen-4](citation_review.md#research_infinigen-4), [market-1](citation_review.md#market-1), [market-3](citation_review.md#market-3), [market-4](citation_review.md#market-4), [market-6](citation_review.md#market-6), [stakeholder-3](citation_review.md#stakeholder-3), [stakeholder-6](citation_review.md#stakeholder-6)

**사람 검수:** 미검수

## gap-domain-1 · domain

**원래 공백:** 기업 IT 사업 문서 검토에서 요구사항·조항 추출, 인용, 요약, 리스크 판정, RAG 및 도구 호출 품질을 두 기술로 검증한 실험이 없다.

**종합 판단:** unresolved

**판단 이유:** 두 논문은 일반 LLM 정확도·장문 처리·메모리·지연을 평가했으나, 기업 IT 사업 문서의 요구사항·조항 추출, 인용, 요약, 리스크 판정, RAG·도구 호출 품질을 두 기술로 검증한 실험은 확인되지 않는다.

**근거 주장:** [domain-1](citation_review.md#domain-1), [domain-3](citation_review.md#domain-3), [domain-4](citation_review.md#domain-4), [domain-6](citation_review.md#domain-6), [stakeholder-1](citation_review.md#stakeholder-1), [stakeholder-4](citation_review.md#stakeholder-4)

**사람 검수:** 미검수

## gap-domain-2 · domain

**원래 공백:** SK AX의 장문·반복·동시 요청은 적용 가정이며, 실제 내부 구조·업무량·도입 또는 채택 사실은 확인되지 않았다.

**종합 판단:** unresolved

**판단 이유:** 기존 주장은 장문·반복·동시 요청을 적용 가정으로만 두고, 실제 기업 데이터 분포·동시성·운영환경과 SK AX의 내부 구조·업무량·도입 또는 채택 사실은 확인하지 않는다. 따라서 목표 업무의 실제 적용 근거 공백은 남는다.

**근거 주장:** [domain-1](citation_review.md#domain-1), [domain-3](citation_review.md#domain-3), [research_kivi-4](citation_review.md#research_kivi-4), [research_infinigen-4](citation_review.md#research_infinigen-4), [market-1](citation_review.md#market-1), [market-4](citation_review.md#market-4)

**사람 검수:** 미검수

## gap-domain-3 · domain

**원래 공백:** KIVI와 InfiniGen을 동일 모델·정밀도·입출력 길이·배치·GPU·CPU 메모리·PCIe 조건에서 비교한 정확도·지연·메모리 실험이 없다.

**종합 판단:** unresolved

**판단 이유:** KIVI의 Llama-2-7B·A100·ShareGPT 효율 실험과 InfiniGen의 OPT-13B·RTX A6000·PCIe 3.0×16 지연 실험은 각각 확인되지만, 동일 모델·정밀도·입출력 길이·배치·GPU·CPU 메모리·PCIe 조건의 정확도·지연·메모리 비교는 없다.

**근거 주장:** [research_kivi-3](citation_review.md#research_kivi-3), [research_infinigen-3](citation_review.md#research_infinigen-3), [domain-6](citation_review.md#domain-6), [market-5](citation_review.md#market-5)

**사람 검수:** 미검수

## gap-domain-4 · domain

**원래 공백:** InfiniGen의 attention prediction 오류율과 PCIe·CPU 메모리 구성별 민감도, KIVI의 기업 문서 근거 보존 손실은 제공 발췌에서 확인되지 않았다.

**종합 판단:** unresolved

**판단 이유:** InfiniGen은 특정 OPT-6.7B 조건에서 alpha·partial weight ratio 민감도와 ratio 0.3 선택은 제시하지만, attention prediction 오류율과 PCIe·CPU 메모리 구성별 민감도는 확인되지 않는다. KIVI의 Falcon·GSM8K 손실 위험은 확인되나 기업 문서의 근거 보존 손실은 검증되지 않았다.

**근거 주장:** [research_infinigen-2](citation_review.md#research_infinigen-2), [research_infinigen-3](citation_review.md#research_infinigen-3), [domain-5](citation_review.md#domain-5), [stakeholder-5](citation_review.md#stakeholder-5), [research_kivi-2](citation_review.md#research_kivi-2), [domain-2](citation_review.md#domain-2)

**사람 검수:** 미검수

## gap-domain-5 · domain

**원래 공백:** 두 기술의 운영 SLA, 장애 복구, 보안·접근통제, 공식 라이선스 적합성 및 독립 재현 절차는 목표 업무 적용 전에 추가 확인이 필요하다.

**종합 판단:** unresolved

**판단 이유:** 두 기술의 공개 구현과 실험실 평가 범위는 확인되지만, 목표 업무의 운영 SLA·장애 복구·보안 및 접근통제·공식 라이선스 적합성·독립 재현 절차는 제공된 주장으로 확인되지 않는다. 따라서 도입 전 검증 공백이 남는다.

**근거 주장:** [research_kivi-4](citation_review.md#research_kivi-4), [research_infinigen-4](citation_review.md#research_infinigen-4), [market-1](citation_review.md#market-1), [market-4](citation_review.md#market-4), [stakeholder-2](citation_review.md#stakeholder-2), [stakeholder-3](citation_review.md#stakeholder-3), [stakeholder-6](citation_review.md#stakeholder-6)

**사람 검수:** 미검수
