# 공식 모델 설정과 별도 기업문서 확보 기록

확인일: 2026-09-21. 모델 성능 측정과 별도로 공식 설정을 확인했다.

| 모델 | 고정 revision | 질문 / 본문 접두사 | 공식 입력 한도 | dense 차원 |
| --- | --- | --- | --- | --- |
| intfloat/multilingual-e5-small | 614241f622f53c4eeff9890bdc4f31cfecc418b3 | `query: ` / `passage: ` | 512 | 384 |
| BAAI/bge-small-en-v1.5 | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a | `Represent this sentence for searching relevant passages: ` / 없음 | 512 | 384 |
| BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 | 없음 / 없음 | 8192 | 1024 |

공식 근거: [E5 고정 카드](https://huggingface.co/intfloat/multilingual-e5-small/raw/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md), [BGE-small 카드](https://huggingface.co/BAAI/bge-small-en-v1.5), [BGE-small tokenizer](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer_config.json), [M3 카드](https://huggingface.co/BAAI/bge-m3), [M3 고정 가중치](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/pytorch_model.bin), [SentenceTransformer 공식 API](https://sbert.net/docs/package_reference/sentence_transformer/model.html).

M3의 벡터 1024차원과 별도 입력 1024토큰 실험은 서로 다른 설정이다. 주 비교는 같은 본문·질문을 쓰고 모델별 prefix·특수 토큰을 포함한 길이를 검사한다. M3 공식 상한8192를 실제 운영 청크 길이로 해석하지 않는다. CPU batch1, 정규화 dense 임베딩, `trust_remote_code=False`를 사용한다. 공식 revision API에서 M3는 safetensors 대신 `pytorch_model.bin`을 제공함을 확인했다. 실제 환경의 torch는 2.14.0이며 원격 custom Python 코드를 실행하지 않는다. 실행 패키지 버전·시간·메모리는 원시 실험 결과에 별도로 기록한다.

## 별도 기업문서 실험: 미확보

공개 IT 제안요청·계약 문서 최대3개60페이지를 별도 corpus로 평가하려 했으나 아래 접근 실패로 원본 PDF를 확보하지 못했다. 웹 도구가 보여주는 추출문을 검증된 로컬 PDF로 대체하지 않는다. 따라서 기업문서16질문(선택8/검증8)과 모델 점수는 만들지 않았고, 기업문서의 검색 품질은 미검증이다.

| 후보 | 확인 범위 | 실제 다운로드 결과 |
| --- | --- | --- |
| [Wilsonville IT Strategic Plan RFP](https://www.wilsonvilleoregon.gov/sites/default/files/fileattachments/it_amp_gis/page/125658/rfp_it_strategic_plan.pdf) | 웹 도구상36페이지, 조건·배점·계약 포함 | 로컬 urllib 및 일반 브라우저 UA curl 모두 HTTP403 |
| [Silver Spring Township IT RFP](https://cms2.revize.com/revize/silverspringtowns/Documents/How%20do%20i/Bids%20Proposals/2018%20SST%20IT%20RFP.pdf) | 웹 도구상12페이지, 현재 공식 입찰목록의 직접 연결 미확정 | 조사자의 실제 HTTP 요청403 |
| [Minnesota Managed IT Services RFP](https://cle.mn.gov/wp-content/uploads/2024/02/RFP-Managed-IT-Services.pdf) | 대안 URL 확인 | 조사자의 실제 HTTP 요청404 |

[Northwest Arctic Borough IT Support RFP](https://www.nwabor.org/wp-content/uploads/RFP-21-02-IT-Support.pdf)는 웹 도구상10페이지이지만 추출문0줄의 스캔형 후보여서 offset 기반 평가 대상에서 제외했다. 별도 재배포 라이선스도 확인되지 않았다. 제품 논문 corpus에는 이 후보들을 넣지 않았다.

재개하려면 접근 가능한 공식 원본 PDF를 확보한 뒤 원문해시·물리페이지·추출문을 고정하고 결과를 보기 전에 독립 질문16개의 정답 구절과 분할을 검수해야 한다. 이번 기술 논문 실험 결과를 기업 RFP·계약서 품질이나 기업 업무에 최적인 모델의 증거로 확대하지 않는다.
