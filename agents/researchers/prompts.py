"""Korean research instructions copied from the legacy pipeline and extended for the contract."""

BIAS_RULES = """\
두 기술 모두 한계, 반대 근거와 부담을 이익과 같은 비중으로 찾고 대칭적인 기준으로 비교하라.
가능하면 서로 다른 출처의 주장을 교차 확인하라. 반대 근거를 찾지 못하면 caveats에
`검색 범위에서 반대 근거 미확인`이라고 쓰고, 반대 근거가 존재하지 않는다고 단정하지 마라.
"""

FACT_INFERENCE_RULES = """\
`source_fact`와 `author_reported_result`는 제공된 청크의 인용문으로 직접 뒷받침되는 내용에만 사용하라.
기업 문서 검토에 적용하는 판단은 `team_inference` 또는 `scenario`로 표시하라.
검색해 제공되지 않은 내용은 `unknown`으로 두고, 검색되지 않은 사실을 보충하거나 만들어 내지 마라.
"""

FEEDBACK_RULES = """\
feedback이 있으면 각 항목을 검색과 분석에 빠짐없이 반영하라.
확인한 내용은 그 근거와 함께 답하고 해결되지 않은 항목은 구체적인 이유와 함께 gaps에 남겨라.
"""

PAPER_QUERY_RULES = """\
사용자 질문은 한국어여도 된다. 논문 검색 query는 질문의 의미와 대상 facet을 유지해 짧은 영어 ASCII 문장으로 바꾸라.
"""

COMMON_RESEARCH_RULES = "\n".join((BIAS_RULES, FACT_INFERENCE_RULES, FEEDBACK_RULES))

RESEARCH_PLAN_PROMPT = """\
각 기술별 mechanism, limitation, conditions, maturity에 대한 영어 논문 검색 질문을 하나씩, 총 8개 작성하라.
conditions는 적용조건이 아니라 논문의 실험환경이다. KIVI는 efficiency Figure 5의 NVIDIA A100과 ShareGPT,
InfiniGen은 section 5.1 experimental setup의 A6000 CPU PCIe models를 찾는 질문으로 작성하라.
mechanism은 핵심 원리, limitation은 정확도·설정의 한계, maturity는 구현과 실증 범위를 찾는다.
각 query는 영어 ASCII 400자 이내, 5~30단어의 짧은 검색문이며 모든 표·세부조건을 나열하지 마라.
제공된 domain, scenario, questions를 검색 맥락으로 반영하되, 질문의 적용 시나리오를 논문상 사실로 취급하지 마라.
""" + PAPER_QUERY_RULES + COMMON_RESEARCH_RULES

RETRIEVAL_REVIEW_PROMPT = """\
기술 조사 역할 안에서 검색 후보의 관련성과 충분성을 평가하라. mechanism, limitation, conditions, maturity
각각 하나씩 총 네 item을 작성하라. 각 질문에 실제로 답하는 chunk_ids와 reason을 적고,
근거가 부족하면 sufficient=false와 구체적인 missing 목록을 적어라.
목표는 기술별 네 개의 짧은 근거 기반 주장이지 논문 전체 재현이나 모든 실험의 복원이 아니다.
최소 충분 기준: mechanism은 핵심 동작과 잔여 캐시/선택 원리, limitation은 원문에 있는 한 가지 구체적 제약,
conditions는 확인된 장비·모델·baseline 등 한 실험의 식별 가능한 설정, maturity는 논문 구현·실험 범위다.
이 최소 근거가 있으면 모든 표·알고리즘 전문·warmup·commit·독립 재현·기업 운용 증거가 없다는 이유만으로 false로 하지 마라.
없는 부가조건과 운영 실증은 평가의 caveats/gaps에 미확인으로 남긴다. 질문이 과도하게 넓어도 최소 기준으로 판단한다.
핵심 원리나 실험설정 자체를 뒷받침할 원문이 없으면 false를 유지한다. reason에 확인 범위와 한계를 설명하라.
sufficient=true이면 missing은 빈 목록이어야 한다. 이는 잠정 판단이며 사실 검증 완료를 뜻하지 않는다.
""" + COMMON_RESEARCH_RULES

RESEARCH_ASSESSMENT_PROMPT = """\
논문을 근거로 이 기술만 평가하라. 정확히 4개 claim으로 mechanism, limitation, conditions, maturity를 하나씩 작성하라.
maturity에는 공개 정보 기반 잠정 TRL 범위와 근거·실증 한계를 적고 kind=team_inference로 표시하라.
TRL 기준은 1 기초원리, 2 개념정립, 3 개념검증, 4 실험실검증, 5 대표 사용조건 검증,
6 관련환경 시스템시연, 7 운용환경 시제품, 8 적격성 검증된 완성시스템, 9 실제 지속운용이다.
논문·코드가 공개됐다는 이유만으로 4에서 5로 높이지 마라. 대표 사용조건의 검증 근거가 있어야 한다.
TRL 숫자를 논문이 직접 인증한 값처럼 쓰지 마라. conditions는 적용 가정이 아니라 논문 실험환경의 사실이다.
conditions claim의 kind는 source_fact 또는 author_reported_result, conditions 필드에 확인된 장비·모델·기준을 적어라.
알려진 공통 장비를 보존하되 서로 다른 수치 실험의 모델·배치·길이를 한 조건으로 합치지 마라.
""" + COMMON_RESEARCH_RULES

RETRIEVAL_REWRITE_PROMPT = """\
이전 검색·평가의 실패 이유를 해결하도록 질문을 다시 작성하라. 같은 기술과 네 facet을 유지하라.
missing_reasons와 이전 후보 내용을 읽고 부족한 핵심 원문을 겨냥하라.
query는 영어 ASCII 400자 이내의 짧은 검색문이다. 부족 항목을 모두 나열한 보고서 작성 지시를 만들지 마라.
한 facet마다 구체적 검색어 5~30단어를 고르고 논문에 없는 운용 실증까지 요구하지 마라.
""" + PAPER_QUERY_RULES + COMMON_RESEARCH_RULES

DEFAULT_WEB_QUERIES = {
    "market": "KIVI InfiniGen adoption deployment alternatives quantization offloading maintenance licensing costs ecosystem",
    "stakeholder": "AI 문서 검토 사용자 신뢰 검토 책임 운영 모니터링 보안 거버넌스 GPU CPU 인프라",
    "domain": "AiPMO RFP 계약 사업 문서 검토 장문 정확도 근거 추적 지연 동시 요청 평가",
}

FACET_DESCRIPTIONS = {
    "market": "adoption(채택 근거와 미확인), alternatives(대체재와 선택 조건), costs(도입·운영 비용)",
    "stakeholder": "user(문서 검토자의 효익·검증 부담), operator(운영자 통합·관측 부담), governance(구매·보안·책임)",
    "domain": "fit(문서 검토 흐름 적합성), risks(정확도·지연·보안 위험), evaluation(검증 실험·판정 지표)",
}

_PERSPECTIVE_INTRODUCTIONS = {
    "market": "단일 도메인 내 구매 이유, 대체 기술, 배포·유지 비용과 생태계 신호를 비교한다. 공개 채택/시장 규모는 미확인이면 명시한다.",
    "stakeholder": "같은 도메인의 문서 검토자, AI 운영자, 구매·보안·인프라 담당자 관점의 이익·부담·충돌을 비교한다. 실제 인터뷰 결과로 쓰지 마라.",
    "domain": "동일한 문서 검토 시나리오에서 두 기술의 적용조건, 정확도·지연·메모리 trade-off, 확인할 실험과 적용 한계를 비교한다.",
}

_PERSPECTIVE_ASSESSMENT_RULES = """\
 양 기술 각각 정확히 3개 claim, 총 6개를 다음 facet별 하나씩 작성하라: {facets}.
기술의 벤치마크를 반복 요약하지 말고 해당 관점의 판단 질문에 답하라.
tech_assessment의 conditions/caveats/references를 보존해서 판단하라. 알려진 공통 장비는 유지하되
서로 다른 수치 실험의 모델·길이·배치 조건을 합치지 마라. 사실 근거와 팀 해석을 구분하라.
conflicts에는 누가 어떤 효과를 얻고 누가 어떤 추가 부담을 맡는지, 둘이 충돌하는 조건을 적어라.
자료가 부족한 facet도 unknown과 caveats로 표시하고 status=insufficient와 gaps를 남겨라.
"""

PERSPECTIVE_PROMPTS = {
    view: introduction + _PERSPECTIVE_ASSESSMENT_RULES.format(facets=FACET_DESCRIPTIONS[view]) + COMMON_RESEARCH_RULES
    for view, introduction in _PERSPECTIVE_INTRODUCTIONS.items()
}

PERSPECTIVE_REWRITE_PROMPT = """\
부족한 facet만 공식 스냅샷 검색용 짧은 질의로 다시 작성하라. facet마다 하나만 작성한다.
""" + COMMON_RESEARCH_RULES

PERSPECTIVE_REASSESSMENT_PROMPT = """\
 이전 평가의 충분한 facet을 보존하고 missing_facets만 새 근거로 재평가하라.
양 기술 각 3개 총6개 claim과 원래 facet을 유지한다. 남은 자료 부족은 unknown/insufficient로 둔다.
일반 논문 실험의 부분 확인과 기업업무 미검증을 구분하라.
선행 tech_assessment에서 확인된 조건을 미확인으로 되돌리지 마라.
실제 장비의 wall-clock 실험이 확인됐다면 그 사실과 반복 횟수·소프트웨어 세부의 미확인을 분리하라.
text/conditions/caveats/gaps의 근거 부족 표현은 제공된 자료·검색 발췌 범위로 한정하고,
'공개 근거가 없다'처럼 전체 공개 자료의 부재로 단정하지 마라.
""" + COMMON_RESEARCH_RULES


__all__ = [
    "BIAS_RULES",
    "FACT_INFERENCE_RULES",
    "FEEDBACK_RULES",
    "PAPER_QUERY_RULES",
    "COMMON_RESEARCH_RULES",
    "RESEARCH_PLAN_PROMPT",
    "RETRIEVAL_REVIEW_PROMPT",
    "RESEARCH_ASSESSMENT_PROMPT",
    "RETRIEVAL_REWRITE_PROMPT",
    "DEFAULT_WEB_QUERIES",
    "FACET_DESCRIPTIONS",
    "PERSPECTIVE_PROMPTS",
    "PERSPECTIVE_REWRITE_PROMPT",
    "PERSPECTIVE_REASSESSMENT_PROMPT",
]
