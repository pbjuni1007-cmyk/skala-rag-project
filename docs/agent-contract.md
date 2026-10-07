# Agent 협업 입출력 계약 v1

팀의 조사·보고서·평가·실행 담당자는 이 문서의 입력과 출력을 기준으로 노드를 연결한다. 계약 버전은 `agent-contract-v1`이다. 기존 역할은 [#1](https://github.com/pbjuni1007-cmyk/skala-rag-project/issues/1) 박병준, [#2](https://github.com/pbjuni1007-cmyk/skala-rag-project/issues/2) 김기현, [#3](https://github.com/pbjuni1007-cmyk/skala-rag-project/issues/3) 김도현, [#4](https://github.com/pbjuni1007-cmyk/skala-rag-project/issues/4) 홍수정으로 유지한다.

2026-10-07 팀 `main`의 `e1cb88825f56b8358376756d3d994dc5a928b8ed`를 기준으로 작성했다. 이 문서는 앞으로 구현할 노드 사이의 계약이다. 현재 `main`에 Supervisor와 아래 응답 형식이 이미 구현됐다는 뜻은 아니다. 개인 실습의 전체 구현을 가져올 필요도 없다.

## 먼저 연결할 것

| 담당 | 제공할 경계 | 먼저 받을 것 |
| --- | --- | --- |
| #1 박병준 | 공통 형식, State, Supervisor 결정과 그래프 연결 | #2 조사 결과, #3 품질 판정, #4 실행·추적 연결 |
| #2 김기현 | `research(request) -> ResearchResult` | `ResearchRequest`와 보완 요청 |
| #3 김도현 | `write_report(request) -> ReportResult`, `evaluate_report(request) -> EvaluationResult` | 네 관점의 조사 결과와 원문·출처 |
| #4 홍수정 | `publish(request) -> PublishResult`, 실행·추적·통합 테스트 | 통과한 보고서와 그 보고서의 평가 결과 |

함수 이름은 연결부에서 사용할 이름이다. 파일 배치·클래스 이름·동기/비동기 구현은 담당자가 정하고 #1에서 연결한다. 하위 노드는 결과만 반환한다. 다른 하위 노드를 직접 호출하거나 그래프의 다음 이동을 실행하지 않는다.

[예시 JSON](agent-contract-examples.json)의 `examples`에서 각 `type`에 해당하는 `payload`를 복사해 가짜 응답으로 개발할 수 있다. 예시의 출처와 주장은 **가상 테스트 데이터**다. 실제 기술 평가·LangSmith 실행·제출 근거로 사용하지 않는다.

## 공통 규칙

모든 요청과 응답에는 다음 다섯 필드를 둔다. 응답은 받은 값을 그대로 돌려준다. 아래 표에서 `?`는 값이 `null`일 수 있다는 뜻이며 필드 자체는 생략하지 않는다. JSON 객체의 키는 문자열이다.

| 필드 | 자료형 | 의미 |
| --- | --- | --- |
| `contract_version` | 문자열 | 반드시 `agent-contract-v1` |
| `run_id` | 비어 있지 않은 문자열 | 한 번의 전체 실행 식별자 |
| `request_id` | 비어 있지 않은 문자열 | 실행 안에서 유일한 요청 식별자. 역할·시도별로 새로 발급 |
| `attempt` | 1 이상의 정수 | 해당 역할의 누적 실행 횟수. 재조사 시 증가 |
| `context` | `RunContext` | 실행 전체에서 고정하는 기술·도메인·적용 시나리오 |

`RunContext = {technologies: ["KIVI", "InfiniGen"], domain: str, scenario: str}`다. `domain`은 적용 업무, `scenario`는 그 업무의 구체적인 가정과 범위다. 둘 다 비어 있지 않은 문자열이다. #4의 실행 진입점이 팀 `config/run.yaml`에서 읽어 #1에 전달하고, #1이 모든 요청에 넣는다. writer는 이 값으로 기술·시나리오 정보를 작성하고 PDF 담당은 `ReportResult.context`를 renderer의 config로 전달한다. 같은 `run_id`에서 context가 다른 결과를 합치지 않는다.

관점 이름은 `research`, `market`, `stakeholder`, `domain`이다. 기술 이름은 `KIVI`, `InfiniGen`을 사용한다. 결정·결과·보고서를 같은 `run_id`로 연결한다. 같은 요청 ID를 다른 내용에 다시 사용하지 않는다. JSON으로 저장할 수 있는 값만 교환하며 모델 객체·API 키·예외 객체를 넣지 않는다.

### 근거 자료형

`Claim`, `Reference`, `Assessment`, `Report`는 기준 커밋의 [rag/schemas.py](../rag/schemas.py) 형식을 재사용한다. 새 wrapper의 필드를 기존 `Assessment`에 섞어 넣지 않는다.

| 자료형 | 필수 필드 |
| --- | --- |
| `Reference` | `chunk_id: str`, `quote: str` |
| `Claim` | `technology: KIVI\|InfiniGen\|both`, `facet: str`, `text: str`, `kind`, `references: Reference[]`, `caveats: str`, `conditions: str` |
| `Assessment` | `status: ok\|insufficient`, `claims: Claim[]`, `conflicts: str[]`, `gaps: str[]` |
| `Chunk` | `id: str`, `source_id: str`, `technology: KIVI\|InfiniGen\|both`, `text: str`, `page: int?`, `section: str?` |
| `Source` | `type: paper_pool\|external_web`, `authors: str`, `title: str`, `url: str`, `version: str`, `date: str?`, `accessed_at: str` |

`Claim.kind`는 `source_fact`, `author_reported_result`, `team_inference`, `scenario`, `unknown` 중 하나다. `grounding` 같은 개인 실습의 추가 필드는 포함하지 않는다. `Chunk`와 `Source`에는 해시 등 기존 수집 메타데이터를 추가로 보존할 수 있다.

`sources`는 `{source_id: Source}` 사전이다. 각 Reference는 `chunks`에 있는 청크를 가리켜야 하고, 각 청크의 `source_id`는 `sources`에 있어야 한다. PDF의 `page`는 1부터 시작하는 물리 페이지이며 `section=null`이다. 웹은 `page=null`이고 `section`에 원문 위치를 적는다. 검색에 쓴 문맥을 요약해 `Chunk.text`를 바꾸지 않는다. 인용문은 원문 청크에서 확인할 수 있어야 한다.

여러 결과의 청크·출처를 합칠 때 같은 ID는 같은 내용을 가리켜야 한다. 내용이 다르면 덮어쓰지 않고 `artifact_mismatch`로 중단한다. 같은 내용은 한 번만 모델 문맥에 넣는다.

`authors`·`accessed_at`이 없는 모델용 `source_metadata()`를 PDF용 `sources`로 넘기지 않는다. 확보하지 못한 저자는 `저자 미표기`처럼 상태를 명시하며 이름을 추측하지 않는다. 수집 시각은 실제 수집 기록에서 가져온다.

`version`은 비어 있지 않은 문자열이다. 원본에서 버전이 없거나 `null`이면 #2의 변환 계층이 `unknown`을 넣는다. 이는 미확인 표시이며 실제 버전 번호가 아니다. 기존 웹 참고문헌 출력이 `version[:12]`를 사용하므로 `null`을 그대로 전달하지 않는다.

## 조사: #1 ↔ #2

### ResearchRequest

공통 다섯 필드에 다음 필드를 더한다.

| 필드 | 자료형 | 규칙 |
| --- | --- | --- |
| `view` | 관점 이름 | 이번에 조사할 관점 하나 |
| `questions` | 비어 있지 않은 문자열 배열 | 이번 조사에서 답할 질문 |
| `feedback` | 문자열 배열 | 처음은 `[]`, 재조사는 부족한 항목과 보완 요청 |
| `previous_result` | `ResearchResult?` | 같은 관점의 직전 결과. 최초는 `null` |
| `research_context` | `ResearchResult?` | 최신 기술 조사 결과. `research` 요청은 `null`, 다른 관점은 `view=research`, `status=ok`인 결과 |

질문은 한국어를 허용한다. 기존 논문 검색의 영어 `Query` 변환은 #2 내부에서 처리한다. 재조사는 같은 질문을 반복하는 대신 `feedback`을 검색과 분석에 반영한다.

### ResearchResult

| 필드 | 자료형 | 규칙 |
| --- | --- | --- |
| `view` | 관점 이름 | 요청과 같음 |
| `status` | `ok\|insufficient\|failed` | 아래 판정 규칙 적용 |
| `assessments` | `{이름: Assessment}` | research는 `research_kivi`, `research_infinigen` 두 키. 나머지는 자기 관점 이름 하나 |
| `chunks` | `Chunk[]` | 결과가 인용하는 원문 청크. 추가 검색 기록은 별도 로컬 로그에 보관 |
| `sources` | `{source_id: Source}` | 청크가 사용하는 출처 |
| `error` | `NodeError?` | 정상·근거 부족이면 `null` |

- `ok`: 모든 Assessment가 `ok`이고 기존 관점별 형식·인용 검사를 통과했다. Supervisor의 근거 충분 판정은 별도로 필요하다.
- `insufficient`: 하나 이상의 Assessment가 `insufficient`이고 부족한 이유가 `gaps`에 있다. 구조가 유효한 부분 결과와 원문은 유지한다.
- `failed`: 검색·파싱·API 등 실행 자체가 실패했다. `assessments={}`, `chunks=[]`, `sources={}`로 반환하고 `error`를 채운다. 부분 실패 기록은 로컬에 보존한다.

`kind=unknown`은 개별 주장의 미확인 상태다. 이를 실행 실패로 바꾸지 않는다. 공개 자료로 확인할 수 없는 도입 사실은 공백으로 남길 수 있다. 기술 원리나 필수 성능 주장의 근거가 없으면 `insufficient`로 반환한다.

기존 연구 결과 `tech_assessment["KIVI"]`는 `assessments["research_kivi"]`로 옮긴다. 다른 관점은 `<view>_result`에서 Assessment 필드만 추린다. 실행 결과에 붙은 `searched_chunks`, `retrieval_diagnostics`는 Assessment 밖에서 처리한다. 기존 연구의 네 facet과 후속 관점의 기술별 세 facet 검사는 유지한다.

## 보고서 작성: #1·#2 → #3

### WriteRequest

공통 필드와 `results: {관점: ResearchResult}`, `feedback: str[]`, `previous_report: ReportResult?`를 받는다. `results`에는 최신 네 관점이 모두 있어야 하며 모두 `status=ok`여야 한다. #1은 별도의 충분성 판단까지 통과한 경우에만 작성 요청을 보낸다. 수정 요청에는 품질 평가가 지적한 이유와 대상 주장 ID를 전달한다.

### ReportResult

| 필드 | 자료형 | 규칙 |
| --- | --- | --- |
| `status` | `ok\|failed` | `ok`는 보고서 생성 완료이며 품질 통과가 아님 |
| `markdown` | `str?` | 완성된 본문. 실패는 `null` |
| `report` | `Report?` | 기존 구조화 보고서. 실패는 `null` |
| `joined` | `Joined?` | 아래의 주장·근거 연결. 실패는 `null` |
| `chunks` | `Chunk[]` | 보고서가 사용하는 원문 청크. 실패는 `[]` |
| `sources` | `{source_id: Source}` | 참고문헌 출처. 실패는 `{}` |
| `error` | `NodeError?` | 성공은 `null`, 실패는 오류 |

`Report`의 필드는 `summary_claim_ids`, `sections: [{title, claim_ids}]`, `synthesis_claims`, `gap_decisions: [{gap_id, status, resolution, claim_ids}]`다. `synthesis_claims`는 기존 근거를 재사용한 `team_inference` 두 개다. 보고서 작성자는 새로운 사실이나 출처를 만들어 추가하지 않는다.

`Joined`는 기존 `Pipeline.join()`의 `claims`, `evidence`, `assessments`, `conflicts`, `gaps`, `gap_records`, `conflict_records`를 보존한다. `claims`와 `evidence`는 각각 ID를 키로 하는 사전이다. `gap_records`는 `{id, perspective, text}` 배열이고, `conflict_records`는 기존 `ConflictRecord` 배열이다. 종합 후의 `claims`·`evidence`에는 종합 주장 두 개와 그 근거도 포함한다.

`collect_evidence()`가 만드는 `research_kivi-1`, `market-1`, `synthesis-1` 등의 주장 ID와 `E-<claim_id>-<순번>` 근거 ID를 사용한다. 한 보고서 안에서는 재번호를 붙이지 않는다. 재조사 후에는 새 결과만 모아 새 보고서를 만들고 이전 보고서의 평가·PDF를 재사용하지 않는다. 주장 ID의 전체 식별 범위는 `(run_id, ReportResult.request_id, claim_id)`다.

`markdown`과 `report`·`joined`는 같은 내용이어야 한다. 사람이 Markdown을 수정했거나 writer가 재작성하면 새 보고서 요청 ID를 발급하고 다시 평가한다. 생성 시점의 구조 오류는 평가 결과에 남길 수 있으나, 참조할 수 없는 출처·청크나 파싱 오류는 `failed`로 반환한다.

## 품질 평가: #3 → #1

`EvaluationRequest`는 공통 필드와 `report_result: ReportResult`를 받는다. 원문이 없는 주장 요약만으로 평가하지 않는다.

| EvaluationResult 필드 | 자료형 | 규칙 |
| --- | --- | --- |
| `report_request_id` | 문자열 | 평가한 ReportResult의 요청 ID |
| `status` | `ok\|failed` | 평가 수행 성공과 오류 구분 |
| `method` | `structural\|llm\|hybrid` | #3이 선택한 평가 방식 |
| `passed` | 불리언 | 네 기준을 모두 통과한 경우만 `true` |
| `checks` | `{기준: Check}` | `groundedness`, `neutrality`, `bias_control`, `coverage` 네 키 |
| `repair_requests` | `RepairRequest[]` | 실패 이유와 보완 대상. 통과는 `[]` |
| `error` | `NodeError?` | 평가 성공은 `null` |

`Check = {passed: bool, reason: str, claim_ids: str[], gap_ids: str[]}`다. `RepairRequest = {target: 관점|writer, reason: str, claim_ids: str[], gap_ids: str[]}`다. ID가 없는 전체 형식 문제는 빈 ID 배열과 구체적인 이유를 반환한다. 있는 ID는 해당 보고서 안에서 해석할 수 있어야 한다.

`status=ok`이면 네 기준이 모두 있어야 한다. 하나라도 미달이면 `passed=false`이고 하나 이상의 보완 요청을 반환한다. `status=failed`이면 `passed=false`, `checks={}`, `repair_requests=[]`와 오류를 반환한다. 평가 오류를 품질 미달이나 통과로 위장하지 않는다.

평가 방식과 내용 검증 범위는 #3이 README에 명시한다. 코드의 출처·인용·보고서 구조 검사가 실패하면 LLM Judge의 긍정 판정으로 덮지 않는다. `structural` 통과를 의미적 정확성 검증으로 소개하지 않는다. #1이 보완 대상을 다음 조사·작성 요청으로 연결한다.

## Supervisor·상태·실패 규칙: #1

`SupervisorDecision`은 공통 필드와 `next_action`, `evidence_sufficient`, `reason_code`, `reason`, `feedback`을 갖는다. `next_action`은 네 관점 또는 `writer`, `evaluator`, `publish`, `stop`이다. `reason`은 비어 있지 않은 설명, `feedback`은 문자열 배열이다. `attempt`는 Supervisor 결정 횟수다.

`reason_code`는 `initial_research`, `missing_view`, `evidence_gap`, `evidence_ready`, `report_ready`, `quality_rework`, `quality_passed`, `limit_exceeded`, `fatal_error` 중 하나다.

LangGraph의 `add_conditional_edges`가 현재 State와 SupervisorDecision으로 다음 노드를 선택해야 한다. `next_action`을 기록만 하고 고정 순서로 실행하지 않는다. 근거 판단과 재조사가 목적이므로 Supervisor를 선택한다. 한 역할씩 실행하면 복구와 판단 경로를 추적하기 쉽지만 병렬 처리보다 느릴 수 있고 조정 호출 비용이 추가된다.

1. research 이후의 관점 순서는 현재 상태로 고른다. `insufficient`를 작성 가능한 결과로 취급하지 않는다.
2. 네 관점의 `ok`와 Supervisor의 충분성 판단이 있어야 writer로 간다. 작성 후 evaluator를 거친다.
3. 근거 보완은 지정 관점으로, 표현·구성 보완은 writer로 보낸다. 하위 역할의 모든 결과는 Supervisor로 돌아온다.
4. research를 다시 실행하면 세 후속 관점과 보고서·평가를 무효화한다. 다른 관점 하나만 바뀌면 그 관점과 보고서·평가만 무효화한다. 누적 시도수는 유지한다.
5. 기본 한도는 조사 역할별 2회, writer 3회, Supervisor 결정 24회다. `attempt`는 최초 실행을 포함한다. 한도 도달은 `incomplete`로 종료하며 성공으로 바꾸지 않는다.

`NodeError = {code: str, message: str, retryable: bool}`다. `code`는 `retrieval_error`, `invalid_response`, `api_error`, `budget_exceeded`, `input_budget_exceeded`, `uncertain_request`, `artifact_mismatch`, `render_error` 중 하나다. `message`는 키·원문·응답 전문을 제외한 짧은 설명이다. 이 계약의 기본값은 `retryable=false`다. 근거 부족에 따른 재조사는 오류 재시도와 별개다. API·예산·불확실한 요청을 Supervisor가 자동 재전송하지 않는다. 기존 Gateway의 비용 예약·제한된 재시도 처리는 내부 책임으로 유지한다.

State에는 `run_id`, 버전, 실행 상태, 다음 작업, 시도수, 근거 요약, 수정 요청, 결과 참조를 둔다. 원문·과거 응답·보고서 전체를 누적하지 않는다. 노드 호출 직전에 참조를 해석해 위 요청 객체를 구성한다. 결과는 실행 폴더의 파일에 저장하고 `{path, sha256}` 참조를 남긴다. 경로는 실행 폴더 기준 상대 경로이며 폴더 밖을 허용하지 않는다.

State 상태는 `running`, `incomplete`, `needs_attention`, `completed`로 구분한다. 작업 전 pending, 완료 후 snapshot을 저장한다. pending 호출의 완료 여부가 불명확하면 `needs_attention`에서 멈춘다. 완료된 작업만 재개하며 코드·자료·설정·결과가 달라지면 새 실행으로 시작한다. 직렬 노드 실행을 기본으로 하므로 State의 병렬 병합은 구현하지 않는다.

State를 정의하는 코드에는 제어·결과 분리와 원문·로그 보관 위치를 주석으로 남긴다. README에는 다음 일곱 설계 근거를 설명한다.

| 가이드 항목 | 이 계약의 설계 기준 |
| --- | --- |
| 제어와 페이로드 분리 | 다음 작업·시도수·상태와 결과 파일 참조를 구분 |
| 관측성 위치 | 결정·이유는 외부 기록, State는 현재 제어 정보 |
| 지속성 비용 | 역할별 최신 요약·참조만 저장하고 원문·과거 결과는 별도 파일 |
| 상관 | 같은 `run_id`·`request_id`로 State와 실행·추적 기록 연결 |
| 재개·복구 | pending, 완료 snapshot, 오류와 누적 시도수 보존 |
| 동시 처리 | 직렬 노드 실행이므로 reducer 불필요. 역할 내부 병렬 호출은 State 직접 쓰기 금지 |
| 종료 보장 | 시도·결정 상한 도달 시 이유와 함께 미완료 종료 |

## Markdown·PDF·추적: #3 ↔ #4

Markdown 순서는 `# SUMMARY`, 기술·시나리오 정보, `# 기술 성숙도`, `# 시장성`, `# 이해관계자`, `# 도메인 적용`, `# 관점 간 상충과 한계`, 마지막 `# REFERENCE`다. SUMMARY 뒤에는 기존 변환기가 경계로 사용하는 `**대상 기술:**` 줄을 유지한다. 본문은 기존 claim ID·인용·조건·한계를 보존한다.

지원 범위는 기존 변환기의 제목, 일반 문단, 굵게, 표와 `<br>`이다. 복잡한 HTML·중첩 목록·Mermaid를 보고서 본문에 넣지 않는다. 기존 `render_pdf_report()`는 Markdown만 받는 함수가 아니라 `render_report()`의 별칭이다. #4는 전달받은 Markdown을 변환하는 경계를 연결하거나, 같은 본문을 재생성하는 기존 renderer를 사용한 뒤 전달된 Markdown과의 일치를 확인한다. PDF를 만들기 위해 모델을 다시 호출하지 않는다.

`PublishRequest`는 공통 필드, `report_result: ReportResult`, `evaluation_result: EvaluationResult`를 받는다. 보고서·평가·요청의 `run_id`가 같고 평가의 `report_request_id`가 현재 보고서와 같아야 한다. `status=ok`, `passed=true`, 네 기준 통과, 보완 요청 없음이 출력 조건이다.

통합 publisher는 실행 폴더의 `snapshot.json`이 가리키는 보고서·평가 원본을 RunStore의 해시로 검증하고, 요청의 두 결과가 그 원본과 완전히 같은지 확인한다. 요청 ID가 같아도 평가 후 내용을 바꾸면 `artifact_mismatch`다. 별도 JSON 게시 명령도 같은 저장 원본이 필요하다. 요청·응답 필드는 v1을 유지한다.

`PublishResult`는 공통 필드와 `status: ok|failed`, `markdown_ref: ArtifactRef?`, `pdf_ref: ArtifactRef?`, `pdf_pages: int?`, `human_review_pending: bool`, `error: NodeError?`를 반환한다. `ArtifactRef`는 위 `{path, sha256}` 형식이다. 성공은 두 파일 참조, 1~10의 페이지 수, `human_review_pending=true`, `error=null`이어야 한다. 실패는 두 참조와 페이지 수를 `null`로 두고 오류를 반환한다. 중간 파일이 존재한다는 이유로 성공 처리하지 않는다.

#4는 SUMMARY 반 페이지, REFERENCE, 총 10페이지 이하와 한글·표·인용 배치를 검사한다. 통합 renderer는 실제 PDF 페이지 수가 10을 넘으면 게시하지 않는다. 자동 `completed`와 사람의 내용·제출 검수 완료를 구분한다.

추적은 `run_id`, `request_id`, 역할, 시도수, 다음 작업, `reason_code`, 판정·오류 분류를 연결한다. 자연어 결정 이유와 보완 요청 전문은 로컬 기록에 보존한다. LangSmith에는 기존 원문 비전송 원칙을 유지하며 #4가 허용 목록에 분류 코드와 안전한 설명을 연결한다. 동적 판단은 분류 코드와 대응하는 로컬 이유로 확인한다. 키·원문·프롬프트·모델 답변을 추적에 넣지 않는다. 실제 모델·리전·workspace·비용은 실행 담당의 설정으로 관리하며 이 계약에서 임의의 예산을 새로 배정하지 않는다.

## 인수와 변경

#2는 정상·근거 부족·실행 실패 ResearchResult를 제공한다. #3은 같은 입력으로 만든 Markdown과 정상·미달·평가 오류 결과를 제공한다. #4는 충분한 근거, 특정 관점 재조사, 품질 실패 후 수정, 상한·불확실한 요청 중단을 통합 검증한다. 각 담당자는 자신의 테스트와 README 설명을 작성한다. 실제 LangSmith 증거와 모의 테스트는 구분한다.

요청·응답 ID 불일치, 없는 청크·출처·주장 ID, `insufficient` 작성 허용, 다른 보고서의 평가 재사용은 연결 단계에서 거부한다. 빈 목록·빈 문자열·`null`을 서로 다른 의미로 사용한다. 문서와 예시가 충돌하면 문서 규칙을 기준으로 예시를 수정하고 관련 담당자에게 알린다.

필드 이름·필수 여부·enum·ID 규칙의 변경은 #1에 변경 이유와 입출력 예시를 남기고, 영향을 받는 #2~#4를 연결한다. 호환되지 않는 변경은 계약 버전을 올린다. 각 작업은 팀 `main`에서 분기하고 PR 대상은 `main`으로 한다. 기존 RAG 보존 브랜치는 변경하지 않는다. 전체 통합 뒤 ai-deslop을 진행하고 영향받은 검증을 다시 실행한다.

## 실습 요구사항과 완료 확인

이 계약은 아래 요구사항을 구현할 수 있는 연결 규칙이다. 실제 코드·실행·제출물까지 준비해야 실습을 완료할 수 있다.

| 요구사항 | 구현·확인 책임과 완료 증거 |
| --- | --- |
| Supervisor 필수 동작 | #1: State에 따른 `add_conditional_edges`, 충분성 판단 후 작성, 특정 관점 재작업. 하위 노드 간 직접 호출이 없는 코드와 분기 테스트 |
| State 설계 7항목 | #1: 위 일곱 근거가 README와 코드에 함께 반영됨 |
| 작성 후 4기준 평가와 Loop | #3의 별도 평가 노드, #1의 실패 대상 연결, #4의 재평가 통합 테스트 |
| 조정 계층·하위 역할 분리 | #1~#4: 위 함수 경계로 모듈을 나누고 실제 디렉터리 구성을 README에 설명 |
| 동적 동작 실증·재현성 | #4: 실제 LangSmith 경로와 같은 run_id의 코드·로컬 기록·보고서. `tracing-1.png`부터 순서대로 캡처 |
| 평가 보고서 | #3의 SUMMARY·REFERENCE·기존 본문 구성, #4의 실제 PDF 10페이지 이하와 배치 검수 |
| GitHub·README·제출 | #1~#4: 기존 RAG 보존 브랜치와 Agent 작업 구분. README에 패턴 선정 이유·trade-off·동적 처리·State·그래프 이미지·사용법·실제 개인 역할을 기재. PM·PL 역할 제외 |

#4가 `Agent_{캠퍼스}_{X반}_{이름1+이름2+...}.zip`을 구성하고 Git 링크·LangSmith PNG·PDF를 같은 실행 기준으로 확인한다. 제출은 가이드의 반별 Slack thread, DAY2 종료 전 조건을 따른다. 실제 날짜·반·참여자는 제출 공지와 대조한다. 계약 문서 게시나 모의 테스트만으로 동적 실행의 실증을 대신하지 않는다.

## 기존 private 작업과의 차이

팀의 기준은 위 `main` 커밋이다. private RAG의 검색·인용·수정 결과가 전부 팀 저장소에 반영됐다고 전제하지 않는다. 기존 private RAG는 기술 조사 후 세 관점을 고정 병렬 처리하고 종합한다. 이번 Agent는 Supervisor가 다음 작업을 고르고, 보고서 생성 뒤 품질 미달에 따라 다시 조사하거나 작성한다.

private에 추가한 `Grounding`·`GroundedClaim`은 주장과 원문이 같은 대상·실행 단계를 설명하는지 확인하는 정보다. 현재 팀 `Claim`에는 그 필드가 없다. 이 계약도 기본 Claim을 유지하므로 private의 강화된 근거 검사까지 자동으로 포함하지 않는다. 그 검사를 재사용하려면 #2·#3이 처리할 필드와 검사 범위를 #1에서 함께 갱신해야 한다. 기본 계약으로 가이드의 네 평가 기준을 구현할 수 있지만 private와 동일한 검수 강도를 보장하지는 않는다.

별도로 만든 개인용 Supervisor Agent는 이 계약과 같은 큰 흐름을 갖는다. 다만 개인 구현의 `WorkerResult`, `EvidenceSummary`, 저장 방식과 Hybrid 고정 선택을 팀 계약으로 강제하지 않는다. 팀 계약은 기존 Assessment를 감싼 결과와 담당별 함수 경계를 사용하며 평가 방식은 #3이 명시한다.
