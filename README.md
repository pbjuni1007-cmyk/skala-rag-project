# Subject

**KV Cache 다관점 평가 에이전트 — KIVI·InfiniGen 비교**

Agent 협업의 입력·출력과 연결 규칙: [계약 v1](docs/agent-contract.md) · [가상 응답 예시](docs/agent-contract-examples.json). 역할과 완료 기준은 이슈 #1~#4를 따른다.

## Agent Supervisor — 이슈 #1

박병준 담당 [이슈 #1](https://github.com/pbjuni1007-cmyk/skala-rag-project/issues/1)은 공통 자료형, State와 Supervisor를 구현한다. 조사·보고서·평가·출력 담당자가 제공한 함수를 연결하며, 기존 `app.py`의 RAG 실행은 아래 Overview 이후에 설명한다.

Supervisor는 현재 수집된 관점과 근거를 보고 다음 작업을 고른다. 기술 조사 이후 세 관점의 순서를 고정하지 않는다. 모든 관점이 `ok`여도 Supervisor가 근거 충분성을 명시적으로 승인해야 작성할 수 있다. 품질 평가가 보완 대상을 반환하면 해당 관점이나 writer에 이유·주장 ID·공백 ID를 전달한다. 하위 역할은 다른 하위 역할을 호출하지 않으며, 성공·실패·출력 결과 모두 Supervisor로 돌아온다.

근거 충분성 판단과 선택 재조사를 직접 표현할 수 있어 Supervisor를 선택했다. 직렬 실행은 상태 갱신과 재개를 단순하게 만들지만 관점 병렬 처리보다 느릴 수 있다. 추가 모델 호출은 기존 Gateway의 입력·비용 제한을 따른다. 호출 횟수를 고정해 보고서로 넘기지 않는다.

```mermaid
flowchart TD
    START --> S[Supervisor]
    S -->|선택한 조사·보완 대상| W[research / market / stakeholder / domain]
    W --> S
    S -->|네 관점 ok + 근거 충분 승인| R[writer]
    R --> S
    S -->|새 보고서| E[evaluator]
    E -->|판정 + 보완 대상| S
    S -->|현재 보고서의 네 기준 통과| P[publish]
    P --> S
    S -->|완료 / 오류 / 상한| END
```

실제 컴파일한 그래프는 [supervisor-graph.mmd](docs/supervisor-graph.mmd)에 있다. 분기 구현은 [agents/supervisor.py](agents/supervisor.py)의 `add_conditional_edges`를 사용한다. [LangGraph 공식 Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)의 상태·조건부 분기 규칙을 따른다.

### 연결할 모듈과 함수

| 모듈 | 책임 |
| --- | --- |
| `agents/contracts.py` | 공유 계약 v1의 요청·응답 형식, 공통 식별자와 근거 검증 |
| `agents/state.py` | 원문을 누적하지 않는 State와 실행 상한 |
| `agents/supervisor.py` | 허용 작업 판단, 팀 함수 호출, 선택 재작업과 그래프 연결 |
| `agents/store.py` | 실행 잠금, 원자적 snapshot, 변경 불가 결과 파일과 해시 확인 |
| `agents/decision.py` | 기존 Gateway로 판단을 요청하는 `GatewayDecider` |

네 하위 함수는 JSON 사전을 받아 해당 계약의 JSON 사전 또는 Pydantic 모델을 반환한다. 동기 함수 경계를 사용한다. 비동기 노드는 실행 담당이 동기 연결 계층을 준비해야 하며 coroutine을 그대로 반환하지 않는다.

```python
from agents.contracts import RunContext
from agents.decision import GatewayDecider
from agents.state import Limits
from agents.store import RunStore
from agents.supervisor import Nodes, Supervisor

# 각 담당자가 제공한 함수를 연결하는 예시다.
# research: #2, write_report/evaluate_report: #3, publish/gateway/config: #4.
nodes = Nodes(research, write_report, evaluate_report, publish)
coordinator = Supervisor(
    nodes, GatewayDecider(gateway), RunStore(run_directory),
    Limits(worker_attempts=2, writer_attempts=3, supervisor_steps=24),
)
context = RunContext(
    technologies=["KIVI", "InfiniGen"],
    domain=config["domain"], scenario=config["scenario"],
)
state = coordinator.run(run_id, context, identity=execution_fingerprint)
```

`execution_fingerprint`는 #4가 하위 노드 코드, 모델·설정, 고정 자료와 비용 장부를 묶어 만든 식별값이다. Supervisor는 자신의 코드·프롬프트·잠금 파일, context, 실행 상한과 run ID를 자동으로 결합한다. 값이 달라지면 새 실행 폴더를 사용한다. 키를 식별값이나 로그 본문에 넣지 않는다.

요청에는 `contract_version`, `run_id`, `request_id`, `attempt`, `context`가 공통으로 들어간다. 응답은 이를 그대로 돌려줘야 한다. 출처·청크 ID 충돌, 위조 인용, 다른 보고서의 평가, 원래 주장·근거·공백의 변경은 연결 단계에서 거부한다. 생성된 보고서의 구조 문제는 평가 대상이 될 수 있지만 긍정 평가가 기존 구조 검사 실패를 덮을 수는 없다.

### State 설계 근거

| 가이드 항목 | 구현과 이유 |
| --- | --- |
| 제어와 페이로드 분리 | State에는 다음 작업·시도수·현재 상태·작은 요약·파일 참조를 둔다. 원문과 완전한 결과는 호출 경계에서만 읽는다. |
| 관측성 위치 | 결정 전문과 이유는 로컬 `artifacts/supervisor/`와 `events.jsonl`에 남긴다. `on_event`에는 ID·역할·시도·상태·결정 분류만 전달한다. |
| 지속성 비용 | 역할별 최신 참조만 snapshot에 저장하고 크기를 64 KiB로 제한한다. 과거 결과는 별도 파일이다. 실행 파일 수도 역할별 상한으로 제한하며 자동 삭제하지 않는다. |
| 상관 | `run_id`는 snapshot·요청·응답·이벤트를 연결한다. `request_id`는 역할·시도별로 다르다. |
| 재개·복구 | 호출 전에 pending과 누적 시도수를 저장하고 완료 후 snapshot을 교체한다. 완료 결과의 해시를 검증하고 이어 간다. 완료 여부가 불명확한 pending은 재호출하지 않는다. |
| 동시 처리 | 한 역할씩 실행하므로 reducer가 필요 없다. 같은 실행 폴더의 다른 프로세스는 잠금으로 거부한다. 역할 내부 병렬 작업도 State를 직접 수정하지 않는다. |
| 종료 보장 | 조사 역할별 2회, writer 3회, Supervisor 결정 24회가 기본 상한이다. 상한은 성공 조건이 아니며 `incomplete`로 끝난다. |

재개는 같은 인자로 `resume=True`를 지정한다. `pause_after=N`은 이번 호출에서 그래프 노드 N개를 완료한 뒤 멈추는 검증용 옵션이다. `needs_attention`은 확인되지 않은 호출이 있다는 뜻이다. 실행 오류의 `retryable` 표시만으로 Supervisor가 API·예산·불확실한 요청을 다시 보내지 않는다. 실행 잠금은 macOS/Linux의 `fcntl`을 사용한다.

research가 바뀌면 세 후속 관점과 보고서·평가·출력 참조를 무효화한다. 다른 관점 하나만 바뀌면 나머지 관점은 보존하고 보고서 이후를 무효화한다. 시도수는 초기화하지 않는다. 복수 보완 요청의 원문과 ID는 별도 평가 파일 참조로 보존해 첫 관점 수정 후에도 나머지 요청을 잃지 않는다.

### 검증과 다음 연결

```bash
uv run pytest -q tests/test_team_contracts.py tests/test_team_supervisor.py tests/test_team_store.py
```

테스트는 계약 문서의 가상 응답을 사용한다. 서로 다른 조사 순서, 근거 부족 재조사, 보고서·관점 재작업, 오류·상한, 중단 후 재개, 결과 변조와 경로 이탈을 검증한다. 모의 publisher가 만드는 파일은 실제 평가 PDF가 아니다.

#2·#3·#4의 함수와 연결한 실제 모델 실행, PDF 조판·실제 페이지 수 검사, LangSmith 수신·화면 캡처는 아직 남아 있다. Supervisor는 출력 파일의 해시와 Markdown 일치 및 선언된 페이지 상한을 확인한다. 실제 PDF 내용 검수는 #4가 수행한다. `completed`도 사람의 의미·제출 검수 완료를 뜻하지 않는다.

추적 연결은 `Supervisor(..., on_event=handler)`를 사용한다. hook 실패는 로컬 `telemetry_error`로 기록하며 업무 실행을 중단시키지 않는다. 기본 LangGraph 자동 추적은 State·요청 전문 노출을 막기 위해 실행 중 비활성화한다. 기존 `rag.tracing.safe_metadata()`는 새 결정 분류를 아직 허용하지 않으므로 #4가 명시적인 허용 목록으로 연결해야 한다.

## Overview

기업 IT 사업 문서 검토를 지원하는 Agentic AI를 적용 시나리오로 삼아 **KIVI와 InfiniGen의 선택 조건을 비교하는 보고서 생성기**입니다. KIVI는 KV 캐시를 양자화해 저장량을 줄이고, InfiniGen은 CPU의 KV 중 필요한 항목을 GPU로 가져옵니다. 논문과 공식 자료를 검색해 기술 성숙도·시장성·이해관계자·도메인 적용을 평가합니다.

[설계서](docs/design-report.md) · [평가 보고서](reports/latest/report.md) · [인용 검수표](reports/latest/citation_review.md) · [실행 검증](docs/validation.md)

- **Objective:** KIVI와 InfiniGen을 기술 성숙도·시장성·이해관계자·도메인 적용 관점에서 비교하고, 조건에 따른 선택 근거와 상충을 설명합니다.
- **Method:** LangGraph 기반 Multi-Agent + Agentic RAG. 기술 조사 결과를 공유하고 세 관점을 병렬 평가한 뒤 종합합니다.
- **Tools:** 논문·공식 웹 자료 수집, 로컬 임베딩 검색, 원문 인용 검증, Markdown·PDF 보고서 생성, 선택적 LangSmith 추적을 사용합니다.

## Selected Technologies

- **SW — KIVI:** 키와 값의 특성에 맞는 저비트 양자화로 KV 캐시 저장량을 줄입니다. 메모리 절감과 정확도·연산 부담의 관계를 평가하기 위해 선정했습니다.
- **HW — InfiniGen:** CPU·GPU 메모리 계층과 전송 경로를 활용해 필요한 KV를 선택적으로 가져옵니다. 전용 칩이 아니라 하드웨어 자원을 활용하는 시스템 기법이며, 저장량을 줄이는 KIVI와 다른 접근을 비교합니다.

## Features

- **자료 기반 평가:** KIVI·InfiniGen 논문과 공식 웹 자료에서 원리·한계·실험 조건을 수집합니다.
- **관점별 평가와 통합:** 기술 조사 근거를 공유하고 시장성·이해관계자·도메인 평가를 종합합니다.
- **확증편향 방지:** 장점과 한계, 반대 근거를 함께 검토하고 논문 결과와 업무 적용 추론을 구분합니다. 부족한 근거는 재검색하고 남은 공백을 보고서에 표시합니다.
- **근거·상충 검수:** 주장마다 출처와 원문 인용을 연결하고 관점 간 상충을 별도 검수 기록으로 남깁니다.
- **실행 관리와 출력:** 입력·비용 한도, 호출 병렬화, 검색 설정 회귀 검사를 적용하고 Markdown과 PDF를 함께 생성합니다.

### 자료와 검색 실험

KIVI 15페이지와 InfiniGen 18페이지를 물리 페이지 단위로 추출합니다. PDF 처리 상한은 총 200페이지입니다. 출처별 버전·원문 해시·추출문과 페이지를 보존합니다. 공식 웹은 등록된 시작 URL의 허용 host/path 안에서 깊이 1, 시작점별 최대 3건·전체 6건으로 확장합니다. 문서 5MB·전체 20MB·요청 30초를 제한하며 리디렉션도 검사합니다. 자료 수집이 끝나면 실행 중에는 고정 사본을 검색합니다.

임베딩은 CPU에서 `intfloat/multilingual-e5-small`을 실행합니다. `query:`와 `passage:` 접두사, 384차원 정규화 벡터, 코사인 유사도를 사용합니다. 현재 청크는 300토큰·겹침 50토큰, 질의별 상위 후보는 5개입니다. 두 논문은 175개 청크로 구성됩니다.

[E5·BGE 비교 실험](docs/retrieval-experiments.md)에서 E5 300/50을 선택했습니다. 2,000 E5토큰 문맥 기준 등록 구절 완전 회수는 선택용 16문항에서 14/16, 마지막 검증 8문항에서 7/8이었습니다. 이는 등록한 원문 구절의 회수율이며 모든 질문의 의미적 정답률이나 기업 문서의 검색 성능을 뜻하지 않습니다. [초기 E5·MiniLM 비교](docs/embedding-benchmark.md)도 별도 기록으로 보존합니다.

[운영 문맥 진단](docs/context-retrieval.md)에서는 네 질문을 합친 연구 문맥과 후속 관점의 인용 전달을 따로 측정했습니다. 표·그림 설명과 같은 출처의 실험 설정 청크를 보완합니다. 실제 질문 묶음에서 누락된 핵심 조건을 회수했지만, 질문과 인용 구성이 달라지면 일부 조건이 문맥 예산에서 밀립니다. 이미 확인한 실패 사례를 이용한 진단이며, 미회수 구절과 정책별 차이를 원시 결과에 기록합니다.

### 평가 결과 검수

인용 구절이 원문에 존재하는지와 그 구절이 주장 전체를 뒷받침하는지는 다른 질문입니다. 자동 검사는 구절·출처·페이지 연결, 주장 누락·중복, 본문 순서와 요약 개수를 확인합니다. 팀은 인용 검수표에서 실험조건과 해석을, 공백 검수표에서 해소 판정을 확인합니다.

공개 자료로 확인한 실험과 기업 업무에 대한 적용 가정을 구분합니다. 이 프로젝트는 KIVI·InfiniGen 자체의 성능을 재현하지 않으며, 서로 다른 실험 수치로 우열을 단정하지 않습니다. 최종 PDF의 SUMMARY 반 페이지, 한글·표·페이지 배치도 별도로 확인합니다.

## Tech Stack

| 구분 | 사용 기술과 설정 |
| --- | --- |
| Framework | Python 3.11~3.13, LangGraph |
| LLM / Generator | OpenAI `gpt-5.6-luna`; 질의·수정 `low`, 기술·관점 평가 `medium`, 최종 종합 `max` |
| LLM / Judge | 별도 Judge 모델 없이 역할별 평가·재평가와 코드 기반 구조·인용 검증을 결합하며, 의미 검수는 사람이 수행 |
| Retrieval | 별도 Vector DB 없이 NumPy 기반 로컬 벡터 검색·코사인 유사도; 공식 웹 자료는 어휘 검색 |
| Embedding | `intfloat/multilingual-e5-small`, CPU, 384차원; 청크 300토큰·겹침 50·검색 상위 5개 |
| 검색 평가 | 운영 설정의 2,000토큰 문맥에서 등록 근거 완전 회수: 선택 14/16, 최종 검증 7/8. 측정 정의와 원시 결과는 위 검색 실험 기록 참조 |
| Output / Tracing | Markdown·PDF, 주장·인용·상충 JSON 및 검수표, 선택적 LangSmith |

## Agents

| 역할 | 판단과 결과 |
| --- | --- |
| 기술 조사·성숙도 | 기술별 원리·한계·실험조건·잠정 TRL을 정리하고 공통 근거를 전달 |
| 시장성 | 기술별 채택 동기·대안·도입 및 운영 비용 비교 |
| 이해관계자 | 문서 검토자·운영자·구매 및 보안 담당자의 효익과 부담 비교 |
| 도메인 적용 | 같은 문서 업무의 적합 조건·위험·검증 실험 제안 |
| 종합·보고서 | 관점별 주장을 배치하고 이익과 부담의 상충, 선택 조건과 근거 공백 정리 |

## Architecture

```mermaid
flowchart LR
 P[자료 수집·고정] --> R[기술 조사·성숙도]
 R --> M[시장성]
 R --> S[이해관계자]
 R --> D[도메인 적용]
 M --> J[결과 합류]
 S --> J
 D --> J
 J --> Y[종합·보고서]
 Y --> V[구조·인용 검증]
 V --> O[Markdown·PDF·검수표]
```

LangGraph가 선행 조사, 세 관점의 병렬 평가, 합류와 종료를 제어합니다. 기술 조사는 두 기술을 최대 2개, 후속 평가는 최대 3개 동시 호출로 처리합니다. 각 역할은 질문·입력·응답 구조·검증 규칙을 따로 사용하고 모델 호출과 예산 관리는 공유합니다.

기술 조사는 논문을 **벡터 검색**합니다. 후속 세 관점도 관점별 질의로 고정된 공식 웹 자료를 **어휘 검색**하고 공통 논문 근거와 함께 사용합니다. 부족한 이유를 바탕으로 질의를 한 번 수정하는 검색·재평가 절차가 있습니다. 실제 사내 문서를 검색하거나 RFP를 검토하는 서비스는 이 보고서 생성기의 구현 범위에 포함되지 않습니다.

### 조사 에이전트 (#2)

[협업 계약](docs/agent-contract.md)과 [요청·응답 예시](docs/agent-contract-examples.json)에 맞춰 `ResearchAgent(pipeline).research(request)`가 요청 하나의 관점만 처리합니다. `research`는 `research_kivi`·`research_infinigen`, 후속 `market`·`stakeholder`·`domain`은 자기 관점 키의 Assessment를 반환합니다. 결과 상태는 `ok`, 근거 공백을 보존하는 `insufficient`, 실행 실패인 `failed`입니다. 실패 코드는 `retrieval_error`, `invalid_response`, `api_error`, `budget_exceeded`, `input_budget_exceeded`, `uncertain_request`, `artifact_mismatch`, `render_error`로 제한합니다.

최초 기술 조사는 질의 계획 1회 후 두 기술을 최대 2개 작업자로 병렬 검색합니다. 기술마다 검색은 최대 2라운드입니다. 첫 라운드의 불충분한 검색 검토나 구조 검증 실패에 대해 질의를 한 번 고쳐 재검색하며, 각 라운드에는 검색 검토가 한 번, 검토가 충분하면 Assessment 시도가 최대 한 번 있습니다. 따라서 기술별 최대 2회까지 평가를 시도할 수 있습니다. `feedback`과 `previous_result`가 있으면 대상 기술마다 네 facet의 질의를 다시 작성한 뒤 같은 검색 한도를 적용합니다. 이전 결과에서 `ok`가 아닌 기술만 재검색하고, 두 기술이 모두 `ok`면 둘 다 다시 조사하며 대상이 아닌 기술의 기존 평가·인용 청크는 보존합니다. 이전 결과가 `failed`면 새 질의 계획으로 시작하되 feedback을 반영합니다.

최초 후속 조사는 관점별 Assessment를 한 번 시도하고, 미확인 facet이 남으면 질의 재작성·재검색·재평가를 최대 1회 수행합니다. 이전 결과와 feedback을 받은 경우 기존 논문 인용 근거가 유지되면 1~3개 facet의 검색 질의를 다시 만들고 한 번 재평가하며 나머지 facet의 주장은 보존합니다. 질의별 웹 결과는 최대 6개입니다. 기존 논문 근거가 달라졌으면 전체 후속 조사를 다시 수행하고, 이전 결과가 `failed`면 feedback을 넣은 최초 조사로 진행합니다. 구조화 호출은 한 차례의 수정 라운드를 허용하며 독립 수정 단위는 최대 8개입니다.

논문은 기술별 네 facet(`mechanism`, `limitation`, `conditions`, `maturity`)을 `top_k=5`로 E5 벡터 검색하고, facet마다 검색 질의를 하나씩 사용합니다. 문맥은 6,500 E5 토큰이며 이웃 페이지·표·실험 설정 보완에 최대 2,800 토큰을 배정합니다. 후속 관점은 고정 웹 스냅샷을 어휘 검색해 관점 질의와 사용자 질문 질의에서 각각 최대 8개 후보를 모으고 2,000 토큰으로 제한합니다. 기술 조사에서 인용한 근거도 6,500/2,800 토큰 정책으로 전달합니다. 최초 후속 조사에서 facet 재검색은 확장 스냅샷만 대상으로 하고, feedback 재조사는 전체 스냅샷에서 질의별 최대 6개를 가져옵니다. 보강 문맥은 최대 2,000 E5 토큰입니다. 설정의 전체 설명은 위 [자료·검색 실험](#자료와-검색-실험)과 [검색 회귀 검증](docs/retrieval-regression.md)을 참고하세요.

임베딩 설정은 `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, CPU, 정규화한 384차원 벡터, `query:`·`passage:` 접두사입니다. 청크는 300토큰에 50토큰이 겹칩니다. 질의와 청크는 인코더 한도를 넘으면 자르지 않고 실패 처리합니다. 자료 풀은 논문 2편(KIVI 15페이지, InfiniGen 18페이지; 총 33페이지, 상한 200)과 등록 웹 출처 4개입니다. 허용된 경로에서 깊이 1, 출처별 최대 3개·전체 최대 6개를 확장하며, 원문 해시가 확인된 사본을 고정해 검색합니다. 생성 중 실시간 웹 검색이나 내부 문서 검색은 하지 않습니다.

두 기술에는 같은 질문 구조와 facet을 적용하고, `limitation`·비용·위험 등 반대 근거와 부담도 이익과 함께 찾습니다. 반대 근거를 못 찾은 경우에는 `검색 범위에서 반대 근거 미확인`으로 적고 부재를 단정하지 않습니다. `balance_findings`는 모든 관점의 주장 caveat를 확인하고, 기술 조사 `limitation`, 시장성 `costs`, 도메인 `risks`를 반대 근거 facet으로 점검합니다. 이해관계자에는 지정된 단일 반대 facet이 없습니다. 필요한 caveat나 반대 facet이 빠지면 gap을 추가하고 상태를 `insufficient`로 바꿉니다. `balance.json`과 검색 로그에는 관점·기술별 distinct source 수, 최다 출처 인용 비중, 재조사 대상·질의와 검색 결과 ID를 기록합니다. `source_fact`와 `author_reported_result`는 인용 청크로 뒷받침되는 사실에만 사용합니다. 업무 적용 판단은 `team_inference` 또는 `scenario`, 검색 범위에서 확인되지 않은 내용은 `unknown`으로 표시합니다.

결과와 로컬 로그는 요청별 `outputs/<실행ID>/research/<관점>/<request_id>/` 아래에 저장합니다. `result.json`은 매 요청에 기록하고 실패 결과에는 `error.json`도 남깁니다. 기술 조사는 `retrieval/research.json`과 요청 디렉터리의 `balance.json`, 후속 관점은 `retrieval/<관점>.json`과 `retrieval/balance.json`에 검색·균형 진단을 기록합니다.

[ResearchResult 예시](tests/fixtures/research/README.md)는 저장된 레거시 실행 `20260922T053015-c4b5d5`에서 변환한 자료이며 새 에이전트의 실시간 실행 결과가 아닙니다. 계약·인용·검색 흐름 테스트는 다음 명령으로 실행합니다.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_research_*.py
```

## Directory Structure

```text
├── app.py             # 실행 진입점
├── rag/               # Agent 그래프·검색·모델 호출·근거 검증·보고서 출력
├── config/            # 자료·실행·보고서 설정
├── data/              # 논문 추출문·공식 웹 자료 사본
├── .cache/            # 실행 시 준비하는 모델·검색 인덱스
├── eval/              # 검색 질문·정답 구절·회귀 기준
├── scripts/           # 검색 실험·진단 도구
├── experiments/       # 공개 실험 결과
├── tests/             # 회귀 테스트
├── outputs/           # 실행별 생성 결과
├── output/pdf/        # 제출 PDF
├── reports/latest/    # 검토·편집한 공유 보고서
├── docs/              # 설계 정본·실험·운영 기록
└── README.md
```

| 경로 | 책임 |
| --- | --- |
| `app.py`, `rag/graph.py`, `rag/schemas.py` | 실행 진입점, 역할별 그래프, 공유 State와 구조화 응답 |
| `agents/researchers/` | 계약형 조사 요청 처리와 기술 성숙도·시장성·이해관계자·도메인 Assessment 생성 |
| `rag/corpus.py`, `rag/source_discovery.py` | 원문 수집·청킹·임베딩·검색·출처 관리 |
| `rag/llm.py`, `rag/budget.py`, `rag/request_budget.py`, `rag/repair.py` | API 호출·입력 한도·비용·제한된 오류 수정 |
| `rag/evidence.py`, `rag/render.py` | 주장·인용·보고서 구조 검증과 Markdown·PDF 작성 |
| `config/`, `eval/`, `tests/` | 실행·보고서 명세, 고정 검색 질문, 회귀 검사 |
| `scripts/`, `experiments/` | 로컬 검색 실험 도구와 측정 결과 |
| `docs/`, `reports/latest/` | 설계·기술 기록과 최신 보고서·검수표 |

## Usage

### 환경 준비와 실행

Python 3.11~3.13과 [uv](https://docs.astral.sh/uv/)를 사용합니다. 모델과 패키지는 최초 준비 시 내려받습니다.

```bash
uv sync --frozen
cp -n .env.example .env.local
# .env.local의 OPENAI_API_KEY와 USD_TO_KRW를 입력합니다.
uv run python app.py --prepare
uv run python app.py
```

`--prepare`는 공개 자료 수집과 로컬 임베딩을 수행합니다. 일반 실행은 GPT API를 사용하므로 비용이 발생합니다. `.env.local`을 읽으며 같은 이름의 셸 환경변수가 우선합니다. 기본 모델은 `gpt-5.6-luna`, 기본 추론은 질의·수정 `low`, 조사·관점 `medium`, 종합 `max`이며 출력 상한은 64,000토큰입니다. 모델·단가는 [가격 확인 기록](docs/pricing.md)과 연결해 검사합니다.

```bash
uv run python -m pytest -q                    # API 없이 회귀 검사
uv run python scripts/evaluate.py            # 준비한 인덱스의 검색 검사
uv run python app.py --render outputs/<실행ID> # 저장한 결과로 Markdown·PDF 재작성
uv run python app.py --resume outputs/<실행ID> # 동일 코드·입력의 완료 응답 재사용
```

코드를 수정한 뒤에는 `--reuse-calls outputs/<실행ID>`를 사용할 수 있습니다. 자료·검색 정책·모델·설정·의존성·공개 보고서 명세가 같아야 하며, 정확히 같은 요청의 정상 완료 응답만 재사용합니다. 변경된 요청은 새로 생성하고 현재 검증을 다시 거칩니다. 조건이 달라졌으면 복구 옵션 없이 새로 실행합니다.

결과는 `outputs/<실행ID>/`의 `report.md`, `citation_review.md`, `gap_review.md`에 저장됩니다. 주장·인용·출처·실행 설정도 JSON으로 보존합니다. `human_review_pending`은 생성과 자동 검증을 마치고 인용 의미 검수를 기다리는 상태입니다. 일반 실행과 `--render`는 같은 보고서 내용으로 `.md`와 `.pdf`를 함께 생성합니다. PDF는 `RAG-Output_<캠퍼스>_<반>_<팀원>.pdf`이며 제출 정보가 없으면 `RAG-Output_review.pdf`로 저장합니다. `--render`는 추가 GPT 호출 없이 저장된 결과를 다시 출력합니다. PDF 생성에 실패하면 실행은 `incomplete`로 종료됩니다. 팀은 생성된 PDF의 의미와 페이지 배치를 검수합니다. 저장 루트는 `RAG_OUTPUT_DIR`로 바꿀 수 있습니다.

`reports/latest/`는 검토·편집한 공유용 사본으로, 새 실행 때 자동 갱신되지 않습니다. 새 결과를 공유할 때는 해당 실행의 보고서와 두 검수표를 함께 검토해 옮기고 `run.json`의 실행 ID와 파일 해시를 갱신합니다. 실행 원본은 `outputs/<실행ID>/`에 보존합니다.

제출용 PDF: [설계서](output/pdf/RAG-Design_판교-7반_김기현+김도현+박병준+홍수정.pdf) · [평가 보고서](output/pdf/RAG-Output_판교_7반_김기현+김도현+박병준+홍수정.pdf). 원본 Markdown: [설계서](docs/design-report.md) · [평가 보고서](reports/latest/report.md). 설계의 최신 내용은 Markdown 정본을 기준으로 확인합니다.

### 입력 한도와 비용

입력은 생성 전에 제공자 API로 계수합니다. 24,000토큰에서 프로토콜 여유 512를 뺀 23,488토큰을 상한으로 두고, 최초 구조화 요청에는 수정 지시 여유 512를 추가로 확보합니다. 초과 입력은 생성 전에 중단합니다. 후속 관점 재평가는 세부 질문별 최대 세 개 요청으로 나눌 수 있으며, 각 요청에 전체 원문을 유지합니다. 분할 요청이 모두 입력 검사를 통과하면 병렬 생성합니다. 공유 Gateway가 전체 생성 호출을 최대 3개로 제한합니다.

인용·주장 필드·보고서 배치 오류는 해당 부분만 수정합니다. 충분한 기존 주장과 원문은 코드가 보존하고, 합친 응답에 전체 검증을 다시 적용합니다. 한 라운드의 독립 수정은 최대 8개이며 모든 수정 입력을 먼저 검사합니다. 같은 ID와 같은 내용의 청크만 중복 제거합니다.

누적 내부 집행 한도는 45,000원, 팀 예산은 50,000원입니다. 호출 전 최대 비용을 예약하고 실제 사용량으로 정산합니다. 시간 초과처럼 청구 여부가 불명확하면 자동 재전송 없이 예약액을 유지합니다. 429 응답만 제한적으로 재시도합니다. **유료 실행은 담당자 한 명의 환경에서 같은 `.local/usage.json` 장부를 사용합니다.** 다른 컴퓨터나 프로그램의 비용은 자동 집계되지 않습니다. 장부를 초기화하거나 실행별로 나누지 않습니다.

### 선택 사항: LangSmith 실행 추적

LangSmith에서 에이전트별 소요 시간과 모델 호출의 추론 수준·대기 시간·토큰 사용량을 확인할 수 있습니다. 기본값은 비활성화입니다. 사용하려면 `.env.local`에 다음 값을 설정합니다. 실제 API 키는 로컬 파일에만 입력합니다.

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=발급받은_키
LANGSMITH_PROJECT=skala-rag-project
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_WORKSPACE_ID=
```

Endpoint는 자신의 LangSmith 리전에 맞게 지정합니다. 여러 workspace에 걸친 키는 Workspace ID도 지정합니다. 셸 환경변수가 `.env.local`보다 우선합니다. 비활성화하려면 `LANGSMITH_TRACING=false`로 설정합니다.

추적 화면의 `rag_run`에 있는 `run_id`가 로컬 출력 폴더의 실행 ID입니다. 그 아래에 조사·시장성·이해관계자·도메인·종합·렌더링 단계와 `generation` 호출이 연결됩니다. `queue_wait_seconds`는 동시 호출 슬롯 대기, `generation_elapsed_seconds`는 모델 HTTP 요청의 소요 시간입니다. 후자는 제공자 내부 대기 시간도 포함합니다. 캐시 재사용은 `cache_reuse`로 표시하며 새 토큰 사용량에 더하지 않습니다. 모델 조회와 입력 토큰 사전 계산은 별도 LLM 생성 호출로 기록하지 않습니다.

문서·검색 질의·프롬프트·모델 답변·보고서 원문은 전송하지 않습니다. 오류도 고정된 분류만 남깁니다. 보고서는 기존대로 Markdown과 PDF로 저장되며, 추적 전송 실패가 모델 재시도를 유발하지 않습니다. 종료 시 전송을 최대 2초 기다리고 계속 대기 중이면 안내를 출력합니다. 네트워크 장애나 전송 대기열 포화 시 일부 추적이 누락될 수 있으므로 비용·실행 결과의 기준은 로컬 기록입니다. `--prepare`와 `--render`는 추적을 시작하지 않습니다.

### 검색 설정 변경과 상충 검수

인덱스 생성 후 유료 모델 호출 전에 검색 회귀 검사를 수행합니다. 현재 설정과 기준 결과가 일치하면 재사용하고, 모델·청크·검색 개수·문맥 예산·검색 코드가 바뀌면 다시 평가합니다. 질문별 원문·조건 회수가 기준보다 낮아지거나 비교할 기준이 없으면 생성을 시작하지 않습니다. 원문이나 정답이 달라진 경우에는 새 기준을 검토해야 합니다. 재현 명령과 검색 개수·질의 재작성 실험 결과는 [검색 회귀 검증](docs/retrieval-regression.md)에 정리했습니다.

실행 결과에는 `conflict_records.json`과 `conflict_review.md`도 생성됩니다. 상충 원문과 관련 주장 후보·조건·근거를 함께 검토할 수 있습니다. 후보 연결은 검증된 관계와 구분하며, 해소 처리에는 검수자·판단 이유·근거가 있는 주장 연결이 필요합니다. 구조 검증은 잘못된 연결과 부당한 해소 처리를 차단하고, 원문의 의미는 팀이 확인합니다.

기업 문서 검토 품질과 KV 구현의 성능은 별도 실험으로 확인합니다. 자료·질문·반복 횟수·지표·판정 기준은 [도메인 검증 계획](docs/domain-validation-plan.md)을 따릅니다. 현재 검색 실험의 결과를 기업 문서 성능이나 KIVI·InfiniGen의 실측 성능으로 해석하지 않습니다.

## Contributors

| 팀원 | 수행 역할 |
| --- | --- |
| 김기현 | README 재현·PDF 형식·발표 정리 |
| 김도현 | PDF 자료 처리·임베딩·검색·출처 도구 |
| 박병준 | 그래프·State·모델 호출·최종 통합 |
| 홍수정 | 원문·질문셋·관점별 주장·조건 검수 |

## Lessons Learned

- 표의 수치와 실험 설정이 다른 페이지에 놓일 수 있습니다. 구절 회수뿐 아니라 조건을 함께 전달하는지 확인해야 합니다.
- 원문을 잘라 입력 한도를 맞추면 비교 기준을 잃을 수 있습니다. 완전한 청크를 선택하고 오류가 있는 부분만 수정하는 방식으로 요청 크기를 관리했습니다.
- 추론 토큰도 출력 한도와 비용을 사용합니다. 본문 없이 종료된 호출을 포함해 사용량과 미정산 예약을 누적해야 합니다.
- 관점별 근거가 충분해도 종합 결론은 약할 수 있습니다. 최종 판단에서 누구의 이익이 누구의 부담으로 이어지는지 연결해야 합니다.
