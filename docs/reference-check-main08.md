# main-08 출처 검토 기록

실제 Writer 결과의 주장 28개, 인용 72개, 공백 결정 28개와 상충 후보 9개를 원문과 대조했다. 서지·인용 위치와 의미·실험조건을 나누어 검토했다. 제출 보고서는 이 결과를 반영한 편집본이다.

## 검토 범위

Writer가 사용한 출처는 KIVI 논문 33회, InfiniGen 논문 31회, AiPMO 5회, InfiniGen README 3회다. 원문 검토에서 KIVI README의 라이선스 표기를 추가 확인해 제출 보고서에는 참고문헌 5개를 실었다. 등록된 추가 출처 6개는 실제 Writer 인용에 사용되지 않았다.

| 확인 항목 | 원문 근거와 반영 내용 |
| --- | --- |
| KIVI 변수 | group size 변화 시 residual length 128 고정, residual length 변화 시 group size 32 고정. group size 128의 성능 저하를 정확히 구분. 물리 p.7·9 |
| KIVI 효율 조건 | Llama-2-7B, A100 80GB, ShareGPT 기반, 평균 입력 161·출력 338, residual 32·128. 물리 p.7–8 |
| KIVI 평가 지표 | CoQA·GSM8K는 exact match accuracy, TruthfulQA는 BLEU. 물리 p.6 |
| InfiniGen Figure 14 | OPT-13B, 입력 1,920·출력 128, batch 20. INT4·H2O도 CPU offloading 비교 구성. 물리 p.9–11 |
| InfiniGen 민감도 | OPT-6.7B, WinoGrande, 입력 1,920·출력 128, batch 8. ratio 실험은 alpha 4 고정. Figure 18의 OPT-13B 블록 분석과 구분. 물리 p.12–13 |
| 시간 측정 | InfiniGen 저자가 실제 경과 시간을 측정한 결과. 독립 재현은 후속 범위. 물리 p.9 |
| 캐시·동시 처리 | KV tensor와 문서 원문을 구분하고, 메모리 부담 감소가 동시 처리 여지를 늘릴 수 있다는 방향을 반영 |
| 라이선스 | KIVI README의 MIT 표기 확인. LICENSE 전문·모델·의존성의 배포 조건 검토는 후속 범위 |
| 성숙도·업무 효과 | TRL 4는 연구 실험에 근거한 팀의 잠정 평가. AiPMO는 업무 시나리오의 출처. 기업 효과와 이해관계자 부담은 팀 해석 |

상충 후보 9개는 자원 절충·역할별 부담·조건 차이에 관한 해석이다. 동일 조건의 기업 업무 직접 비교, 장기 운영, 비용·보안 검증의 공백은 후속 시험 항목으로 유지했다.

## 출처 식별 정보

논문은 지정 버전의 PDF를 사용했으며 아래 페이지는 물리 페이지를 뜻한다. 웹은 2026-09-21 수집본이다. 논문과 저장 원문의 로컬 대조일은 2026-10-07이다.

| 출처 | 버전·범위 | URL |
| --- | --- | --- |
| KIVI | Zirui Liu 외, arXiv 2402.02750v2 | [PDF](https://arxiv.org/pdf/2402.02750v2) |
| InfiniGen | Wonbeom Lee 외, arXiv 2406.19707v1 | [PDF](https://arxiv.org/pdf/2406.19707v1) |
| KIVI README | KIVI authors, 구현·License | [저장소 원문](https://raw.githubusercontent.com/jy-yuan/KIVI/main/README.md) |
| InfiniGen README | SNU Computer Architecture Lab, artifact evaluation | [저장소 원문](https://raw.githubusercontent.com/snu-comparch/InfiniGen/main/README.md) |
| AiPMO | SK AX, 사업 자료 분석 업무 | [공식 설명](https://www.skax.co.kr/ax-services/aipmo) |

아래 SHA-256은 검토한 파일·추출 본문의 식별값이다. 웹 본문 해시는 Git commit을 뜻하지 않는다.

| 검토 대상 | SHA-256 |
| --- | --- |
| KIVI PDF | `df31ef32d71bfb280c533c5db8220cadf5ef42076bf45d82ba4c8da8e50ea5f4` |
| InfiniGen PDF | `267d689a1ded953f076eb93976c0ebeac1ad02029f1f7c9dd1c947aa05d7cb5f` |
| KIVI README 본문 | `baa1095e6edf8263bbf20507f0d1ce444c3cb57d97d5f5677c2ac19c3b934bbf` |
| InfiniGen README 본문 | `f6a08e32c16d3fdbe8839a95775f2b1e2a2690e36e6ee9d8ec683d6c24e89a90` |
| AiPMO 추출 본문 | `6dbb089c1432e38eaf7f5d93f2d7fa2b4a03ef31a35c9878463f4589f6d77997` |

Writer 원본 SHA-256: `d4382ec7170532ae8dc3400d320beee82cd9a545a6ece419ee1dbb03a664a672`. 실행 ID: `agent-20261007-main-08`.
