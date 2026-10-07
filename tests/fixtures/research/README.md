# ResearchResult 샘플

이 폴더의 네 JSON은 저장된 레거시 파이프라인 실행 `20260922T053015-c4b5d5`를 `agent-contract-v1` 형식으로 변환한 자료입니다. 새 ResearchAgent의 실시간 실행 결과가 아닙니다.

논문 청크는 실행 manifest가 가리키는 해시와 일치하는 로컬 RAG 인덱스에서 가져왔고, 웹 청크는 해당 실행의 저장된 노드 결과에서 가져왔습니다. 출처 메타데이터는 실행의 `sources.json`에서 변환했습니다. 이 실행에 사용된 자료는 공개 논문과 공개 웹 스냅샷이며 내부 문서는 포함하지 않습니다.

샘플을 다시 만들려면 저장된 실행 디렉터리와 출력 디렉터리를 지정해 다음 명령을 실행합니다.

```sh
.venv/bin/python scripts/export_research_samples.py outputs/20260922T053015-c4b5d5 tests/fixtures/research
```
