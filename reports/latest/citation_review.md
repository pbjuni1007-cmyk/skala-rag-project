# 인용 검수

각 주장의 전체 본문·실험조건·한계를 원문과 대조하는 검수표입니다. 현재 의미 검수는 대기 중입니다.

## synthesis-1

**기술:** KIVI · InfiniGen · **주장 종류:** 팀 추론

KIVI는 GPU 메모리·배치 여력을 사용자와 인프라 담당자에게 제공하지만 양자화 회귀와 결과 확인 부담을 남긴다. InfiniGen은 KV 전송 오버헤드 절감을 기대하게 하지만 CPU KV pool, alpha·partial weight, 예측 누락 여부 검증이 운영·사용자 부담이 된다.

**조건:** KIVI 2402.02750v2 Figure 5는 ShareGPT 기반 합성 워크로드, Llama-2-7B, 2비트 KIVI와 FP16(16비트) baseline, 평균 입력 161·출력 338토큰, 단일 NVIDIA A100 80GB에서 배치를 메모리 한계까지 늘려 최대 메모리와 throughput을 비교했다. 보고값은 최대 4배 batch와 2.35배∼3.47배 throughput이며 초기·최종 batch, 측정 소프트웨어·반복 횟수와 시뮬레이션 여부는 제공 발췌에서 미확인이다. KIVI 품질 부담은 Falcon-7B Table 3의 16비트·KIVI-2·KIVI-4 비교를 사용했으며 세부 데이터셋·지표는 일부 미확인이다. InfiniGen은 2406.19707v1의 offloading, alpha 선택, OPT-6.7B·입력 1920·출력 128·batch 8·WinoGrande 민감도와 partial weight ratio 0.3을 기준으로 했다. 업무 역할은 반복·정형 검토를 AI가 하고 전문가가 최종 판단하는 조건이다.

**한계:** 역할별 효익과 부담은 논문 실험과 업무 설명을 연결한 팀 해석이다. 기업 문서의 조항·인용·요약 품질, Agent 도구 호출, attention 예측 오류율, 장기 반복 운영, SLA·보안·공식 라이선스는 제공 근거에서 확인되지 않으며 논문 벤치마크 손실을 업무 오류로 환산하지 않았다.

판정: 미검수

- E-synthesis-1-1 | kivi | 물리 페이지 8 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 3: Performance comparison between 16bit, 4-bit per- token quantization, four fake 2bit KV ca

> with similar maximum memory us- age, KIVI enables up to 4× larger batch size and gives 2.35×∼ 3.47× larger throughput.

- E-synthesis-1-2 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> 4bit KIVI is needed to maintain the accuracy, while 2bit KIVI may have a large accuracy drop in this case.

- E-synthesis-1-3 | infinigen | 물리 페이지 6 | number of key tokens required to reach 0.9 varies even for the adjacent query tokens; for instance, the 998th, 999th, 1000th, 1001st, and 1002nd tokens need 172

> InfiniGen, which enables offloading the KV cache with low data transfer overhead.

- E-synthesis-1-4 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> By dynamically adjusting the number of KV entries to prefetch, InfiniGen brings only the necessary amount of the KV cache to the GPU, thereby greatly reducing the overhead of the KV cache transfer.

- E-synthesis-1-5 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> InfiniGen only prefetches the keys and values of the tokens with an attention score larger than the highest attention score minus alpha.

- E-synthesis-1-6 | infinigen | 물리 페이지 13 | 0 5 10 15 FlexGen INT4 H2O InfiniGen Ideal Latency (ms) Attention FFN Data Transfer Prediction 28.0 H2O Figure 18: Latency breakdown of a Transformer block for 

> increasing the partial weight ratio results in higher memory consumption for partial weights and key cache

- E-synthesis-1-7 | aipmo | 물리 페이지 None | snapshot block 17, character 0

> 반복적이고 정형화된 검토는 AI Agent가 수행하고, 최종 판단은 전문가가 보완하는 Human-in-the-loop 구조로 신뢰성을 확보합니다.

## synthesis-2

**기술:** KIVI · InfiniGen · **주장 종류:** 팀 추론

KIVI는 GPU KV 메모리가 병목이고 품질 회귀시험을 통과할 때 도입 후보이며, InfiniGen은 장문·대배치의 CPU–GPU KV 전송이 병목이고 CPU 메모리·PCIe를 확보할 때 후보가 된다. 문서 Agent 품질과 운영지표 검증 전에는 우위를 확정하지 않는다.

**조건:** KIVI 2402.02750v2의 채널별 key·토큰별 value KV 양자화와 MQA/GQA에서 KIVI-4를 권고한 조건을 GPU KV 메모리 병목 판단에 사용하고, LM-Eval의 CoQA exact match·TruthfulQA BLEU·GSM8K exact match를 품질 게이트로 둔다. InfiniGen 2406.19707v1의 후보 조건은 CPU KV pool과 선택적 prefetch, Figure 14의 OPT-13B·입력 1920·출력 128(시퀀스 2048)·batch 20·RTX A6000 48GB·Xeon Gold 6136·DDR4-2666 96GB·PCIe 3.0×16에서 UVM·H2O(KV budget 20%)·FlexGen·FlexGen+INT4와 prefill/decoding latency를 비교한 시스템 실행이다. 해당 latency 실험의 모델 정밀도·데이터셋은 미확인이고, 선행 accuracy는 WinoGrande로 평가됐다. 목표 업무는 RFP·계약서·사업계획서·발주 문서의 조항·요구사항 추출을 대상으로 인용 precision/recall, 요약 사실성, 도구 호출 성공률, p95 지연, 최대 동시성과 GPU·CPU 메모리·전송량을 full-KV baseline과 별도 측정한다.

**한계:** 시장 도입·업무 적합성은 공개 채택이나 생산운영으로 확인된 것이 아니다. KIVI와 InfiniGen의 선행 실험은 모델·정밀도·입출력 길이·배치·GPU·메모리 계층이 달라 직접 우열을 비교할 수 없고, 문서 Agent의 보안·라이선스·SLA도 제공 근거에서 미확인이다.

판정: 미검수

- E-synthesis-2-1 | kivi | 물리 페이지 2 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache per-token quantization per-channel quantization Figure 1: Definition of per-token and per-channel 

> KIVI quantizes key cache per-channel and quantizes value cache per-token.

- E-synthesis-2-2 | expanded_cdd9ac36bdc16180 | 물리 페이지 None | snapshot block 1, character 0

> For multiquery attention or group query attention, since the keys and values are already compressed, we recommend using KIVI-4

- E-synthesis-2-3 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> By dynamically adjusting the number of KV entries to prefetch, InfiniGen brings only the necessary amount of the KV cache to the GPU, thereby greatly reducing the overhead of the KV cache transfer.

- E-synthesis-2-4 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> We run the experiments on a system equipped with an NVIDIA RTX A6000 GPU [44] with 48GB of memory and an Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory. PCIe 3.0×16 interconnects the CPU and GPU.

- E-synthesis-2-5 | infinigen | 물리 페이지 11 | Table 2: Perplexity on WikiText-2 and PTB with 2048 sequence length with or without KV cache memory limits. Lower is better. Scheme OPT-6.7B OPT-13B OPT-30B Lla

> In this section, we refer to H2O (with a KV cache budget of 20%) and 4-bit quantization implemented on top of FlexGen as H2O and INT4.

- E-synthesis-2-6 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> For LM-eval, we adopt CoQA (Exact match accuracy), TruthfulQA (BLEU score), and GSM8K (Exact match accuracy).

- E-synthesis-2-7 | infinigen | 물리 페이지 12 | 0 2 4 6 512 1024 1536 2048 Speedup INT4 H2O InfiniGen 0 2 4 6 6.7B 13B 30B Speedup INT4 H2O InfiniGen (b) Model Size(a) Sequence Length H2O H2O Figure 16: Speed

> The accuracy is evaluated with the WinoGrande task in lm-evaluation-harness.

- E-synthesis-2-8 | aipmo | 물리 페이지 None | snapshot block 15, character 0

> AiPMO는 RFP, 계약서, 사업계획서, 발주 문서 등 다양한 사업 자료를 자동으로 분석해 검토가 필요한 핵심 항목을 식별합니다.

## market-5

**기술:** InfiniGen · **주장 종류:** 팀 추론

InfiniGen: CPU 메모리와 PCIe를 활용하는 장문·대배치 offloading이 병목이면 H2O·FlexGen·INT4 대비 검토 가치가 있지만, GPU 내 KV 양자화만 필요한 환경에는 복잡도가 커질 수 있다.

**조건:** Figure 14의 비교는 OPT-13B, 입력 1920·출력 128, sequence length 2048, batch 20, RTX A6000 48GB, Xeon Gold 6136, DDR4-2666 96GB, PCIe 3.0×16에서 prefill·decoding latency를 FlexGen, UVM, H2O(KV budget 20%), FlexGen+INT4와 비교한 것이다. 측정은 시스템 실행 결과로 제시되며, 모델 정밀도와 데이터셋은 해당 latency 실험에서 미확인이다.

**한계:** speedup은 InfiniGen 논문의 특정 시스템 실측치이며 KIVI와 동일 조건의 직접 비교가 아니다. PCIe·CPU 메모리가 부족하거나 짧은 요청 중심이면 선택 우위는 확인되지 않는다.

판정: 미검수

- E-market-5-1 | infinigen | 물리 페이지 11 | Table 2: Perplexity on WikiText-2 and PTB with 2048 sequence length with or without KV cache memory limits. Lower is better. Scheme OPT-6.7B OPT-13B OPT-30B Lla

> In this section, we refer to H2O (with a KV cache budget of 20%) and 4-bit quantization implemented on top of FlexGen as H2O and INT4.

- E-market-5-2 | infinigen | 물리 페이지 11 | Table 2: Perplexity on WikiText-2 and PTB with 2048 sequence length with or without KV cache memory limits. Lower is better. Scheme OPT-6.7B OPT-13B OPT-30B Lla

> InfiniGen achieves 1.63×-32.93× speedups over the baselines. The performance benefit mainly comes from the significantly reduced amount of KV cache to load from the CPU memory

- E-market-5-3 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> By dynamically adjusting the number of KV entries to prefetch, InfiniGen brings only the necessary amount of the KV cache to the GPU

## research_kivi-1

**기술:** KIVI · **주장 종류:** 출처 사실

KIVI는 키 캐시를 채널별, 값 캐시를 토큰별로 2비트 양자화하고, 스트리밍에 맞지 않는 키 캐시는 토큰 그룹과 잔여 FP 캐시로 분할해 정확도와 처리 효율을 함께 확보한다.

**조건:** 논문 KIVI v2의 Llama/Llama-2, Falcon, Mistral 계열 평가와 자동회귀 추론의 prefill·decoding 구조를 전제로 한다. 키 캐시는 채널별, 값 캐시는 토큰별로 처리하고, 완전한 그룹을 이루지 못한 잔여 캐시는 FP로 유지한다.

**한계:** 이 원리는 KV 캐시 메모리와 디코딩 비용을 줄이는 방법으로 확인되지만, 기업 IT 문서 검토 Agent의 정확도나 검색·도구 호출 품질까지 직접 검증한 결과는 아니다.

판정: 미검수

- E-research_kivi-1-1 | kivi | 물리 페이지 2 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache per-token quantization per-channel quantization Figure 1: Definition of per-token and per-channel 

> KIVI quantizes key cache per-channel and quantizes value cache per-token. The per-token value cache quantization aligns well with the streaming nature of auto-regressive inference

- E-research_kivi-1-2 | kivi | 물리 페이지 2 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache per-token quantization per-channel quantization Figure 1: Definition of per-token and per-channel 

> We only apply group-wise quanti- zation to the grouped key cache and value cache, while the residual key cache and value cache are kept in full precision.

- E-research_kivi-1-3 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> KIVI maintains a full precision KV cache sliding window for the local relevant tokens.

## research_kivi-2

**기술:** KIVI · **주장 종류:** 저자 보고 결과

KIVI는 모델 구조와 설정에 따라 정확도 손실이 커질 수 있다. 특히 이미 KV가 압축된 Falcon의 2비트 설정과 큰 group size는 대표 태스크의 품질 저하 위험을 높인다.

**조건:** KIVI v2 Table 3의 모델·정밀도·비교 기준은 Falcon-7B의 16bit, KIVI-2, KIVI-4이며, 정확도 지표와 세부 데이터셋은 발췌상 일부 미확인이다. Table 5는 Llama2-13B, GSM8K, group size 32·64·128 및 residual length 실험이며, group size 128에서 17.29로 감소했다.

**한계:** 검색 발췌에서 확인된 한계는 Falcon·GSM8K 등 논문 벤치마크에 관한 것이다. 기업 문서 검토의 사실성, 인용 정확도, 장문 반복 요청에서의 손실 크기는 미확인이다.

판정: 미검수

- E-research_kivi-2-1 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> B adopts multi-query attention and only has one head for KV cache, it is already highly compressed compared to Llama- based models. Thus, in Table 3, 4bit KIVI is needed to maintain the accuracy, while 2bit KIVI may have a large accuracy drop in this case.

- E-research_kivi-2-2 | kivi | 물리 페이지 7 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache 0.5K2K4K7K9K11K13K16K18K20K Word Count 0.0 0.11 0.22 0.33 0.44 0.56 0.67 0.78 0.89 1.0 Depth 20K w

> the performance significantly decreases when the group size reaches 128.

- E-research_kivi-2-3 | kivi | 물리 페이지 9 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 4: Performance evaluation of KIVI on various models across a range of benchmarks in LongBenc

> Llama2-13B 32 20.77 64 21.00 128 17.29

## research_kivi-3

**기술:** KIVI · **주장 종류:** 저자 보고 결과

KIVI의 효율 이득은 ShareGPT 기반 합성 서비스 워크로드에서 확인됐다. Llama-2-7B와 A100 80GB 환경에서 FP16 대비 유사 메모리로 배치와 처리량을 비교한 결과다.

**조건:** KIVI v2 Figure 5의 효율 실험은 ShareGPT 기반 합성 워크로드, 평균 입력 161토큰·출력 338토큰, Llama-2-7B, 2비트 KIVI residual length 32·128, FP16 baseline, 단일 NVIDIA A100 80GB를 사용했다. 배치는 메모리 한계까지 증가시켜 비교했으며 시작·최종 배치 수, 측정 소프트웨어와 반복 횟수는 발췌상 미확인이다. 지표는 최대 메모리와 throughput이다.

**한계:** 이는 실제 기업 Agent 서비스가 아닌 합성 워크로드의 단일 GPU 실험이다. 문서 길이·동시 요청·출력 분포가 달라지면 동일한 처리량과 배치 이득을 보장할 수 없다.

판정: 미검수

- E-research_kivi-3-1 | kivi | 물리 페이지 7 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache 0.5K2K4K7K9K11K13K16K18K20K Word Count 0.0 0.11 0.22 0.33 0.44 0.56 0.67 0.78 0.89 1.0 Depth 20K w

> On average, the data set has an input prompt length lprompt of 161 and an output length lgen of 338

- E-research_kivi-3-2 | kivi | 물리 페이지 8 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 3: Performance comparison between 16bit, 4-bit per- token quantization, four fake 2bit KV ca

> the Llama-2-7B model. The hardware here is a single NVIDIA A100 GPU (80GB). As shown in Figure 5, with similar maximum memory us- age, KIVI enables up to 4× larger batch size and gives 2.35×∼ 3.47× larger throughput.

- E-research_kivi-3-3 | kivi | 물리 페이지 8 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 3: Performance comparison between 16bit, 4-bit per- token quantization, four fake 2bit KV ca

> Figure 5: Memory usage and throughput comparison be- tween 2bit KIVI and 16bit baseline.

## research_kivi-4

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI의 공개 근거 기반 잠정 TRL은 4로 판단한다. 공개 구현과 여러 LLM·벤치마크의 실험실 검증은 있으나, 대표적인 기업 IT 문서 Agent 운용환경과 지속 운영까지의 실증은 확인되지 않는다.

**조건:** 판단 근거는 KIVI 논문 v2와 공개 저장소 정보다. 논문은 Hugging Face Transformers 기반 구현, CUDA 역양자화·행렬곱 융합, Triton 커널, Llama/Llama-2·Falcon·Mistral 평가, LM-Eval·LongBench·Needle-in-a-Haystack 실험을 제시한다. 실험실 GPU 조건과 공개 코드까지는 확인되지만 실제 기업 문서 검토 Agent의 관련환경·운용환경 시스템 시연 조건은 미확인이다.

**한계:** TRL은 논문이 인증한 값이 아니라 공개 실험 범위에 대한 팀의 잠정 해석이다. 코드 공개와 다수 벤치마크만으로 대표 사용조건 검증인 TRL 5 이상으로 높이지 않았으며, 라이선스·운영 SLA·다중 GPU 및 Agent 도구 연동 검증은 확인되지 않았다.

판정: 미검수

- E-research_kivi-4-1 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> We provide a hardware-friendly imple- mentation for running KIVI on GPUs. To minimize the overhead, we have fused the dequantization process with matrix multiplication

- E-research_kivi-4-2 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> We use the Hugging Face Transformers codebase and implement the KIVI algorithm upon it.

- E-research_kivi-4-3 | kivi | 물리 페이지 1 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Zirui Liu * 1 Jiayi Yuan* 1 Hongye Jin 2 Shaochen (Henry) Zhong 1 Zhaozhuo Xu 3 Vladimir Braverman

> The source code is available at https://github.com/jy-yuan/KIVI.

- E-research_kivi-4-4 | kivi | 물리 페이지 13 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache B. NIAH Setting We largely follows the passkey retrieval prompt template of Mohtashami and Jaggi (

> We also show result of Needle-in-a-Haystack Test in Figure 4.

## research_infinigen-1

**기술:** InfiniGen · **주장 종류:** 출처 사실

InfiniGen은 다음 레이어의 attention pattern을 미리 추정해 필요한 KV만 GPU로 전송하는 구조로, 오프라인 weight skewing과 CPU KV pool 관리가 전송량 절감의 핵심이다.

**조건:** 근거 문서는 arXiv:2406.19707v1(2024-06-28)이다. 디코딩 중 Layer i−1의 attention input, Layer i의 partial query weight와 partial key cache로 다음 attention pattern을 추정하고, alpha 임계값으로 KV를 동적으로 선택한다. prefill 단계에서 partial weight를 생성하며, CPU 메모리의 KV pool에서 비빈번 토큰을 제거한다.

**한계:** 이는 offloading 기반 생성 추론에서의 KV 전송·관리 메커니즘에 대한 확인이며, 기업 IT 사업 문서 검토 Agentic AI의 실제 워크플로우에서 효과가 검증된 것은 아니다.

판정: 미검수

- E-research_infinigen-1-1 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> At Layer i− 1 of the decoding stage, InfiniGen speculates on the attention pattern of the next layer (Layeri) using the atten- tion input of Layer i− 1, a partial query weight, and a partial key cache of Layer i.

- E-research_infinigen-1-2 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> InfiniGen only prefetches the keys and values of the tokens with an attention score larger than the highest attention score minus alpha.

## research_infinigen-2

**기술:** InfiniGen · **주장 종류:** 저자 보고 결과

InfiniGen은 attention pattern의 반복 간 변화와 레이어·query별 KV 요구량 차이를 처리하지만, alpha와 partial weight ratio가 정확도·지연·메모리 간 절충을 만들며 오류율은 확인되지 않았다.

**조건:** 논문은 partial weight ratio를 높여도 일정 범위 이후 accuracy 차이가 크지 않다고 보고했으며 ratio 0.3을 선택했다. ratio가 두 배가 되면 partial weights와 key cache의 메모리 overhead가 두 배가 된다고 설명한다. alpha와 ratio 민감도는 OPT-6.7B, 입력 1920, 출력 128, batch 8, WinoGrande 조건에서 평가했다.

**한계:** 제공된 발췌에는 attention 예측 오류율, PCIe 세대별 민감도, 장시간 반복 요청에서의 안정성 수치가 없다. ratio 0.3 선택은 논문 실험의 accuracy와 memory overhead 절충이며 보편적 최적값으로 볼 수 없다.

판정: 미검수

- E-research_infinigen-2-1 | infinigen | 물리 페이지 5 | the KV cache size through key/value evictions at runtime within a constrained KV cache budget [37, 78]. However, all the prior works assume the persistence of a

> However, all the prior works assume the persistence of attention patterns across iterations

- E-research_infinigen-2-2 | infinigen | 물리 페이지 6 | number of key tokens required to reach 0.9 varies even for the adjacent query tokens; for instance, the 998th, 999th, 1000th, 1001st, and 1002nd tokens need 172

> The number of key/value tokens required for each layer differs, and each query token demands a varying number of key/value tokens

- E-research_infinigen-2-3 | infinigen | 물리 페이지 13 | 0 5 10 15 FlexGen INT4 H2O InfiniGen Ideal Latency (ms) Attention FFN Data Transfer Prediction 28.0 H2O Figure 18: Latency breakdown of a Transformer block for 

> increasing the partial weight ratio results in higher memory consumption for partial weights and key cache

## research_infinigen-3

**기술:** InfiniGen · **주장 종류:** 저자 보고 결과

InfiniGen은 RTX A6000·PCIe 3.0×16 환경에서 OPT-13B의 2048 길이·batch 20 지연을 FlexGen·H2O·INT4와 비교해 1.63×–32.93× speedup으로 보고했다.

**조건:** 출처 버전은 arXiv:2406.19707v1(2024-06-28)이다. Figure 14의 지연 실험은 OPT-13B, sequence length 2048(입력 1920·출력 128), batch 20, NVIDIA RTX A6000 48GB, Intel Xeon Gold 6136, DDR4-2666 96GB, PCIe 3.0×16에서 수행했고 prefill·decoding latency를 FlexGen, UVM, H2O(KV budget 20%), FlexGen+INT4와 비교했다. 지표는 inference latency이며 시뮬레이션이 아닌 시스템 실행 결과로 제시된다. 모델 정밀도와 해당 latency 실험의 데이터셋은 미확인이다. 별도 정확도 실험은 OPT 6.7B/13B/30B 및 Llama-2 7B/13B, 5-shot COPA·OpenBookQA·WinoGrande·PIQA·RTE와 WikiText-2·PTB를 사용했다.

**한계:** 수치는 논문이 제시한 특정 단일 시스템·모델·길이·배치의 실측 결과이며, 다른 GPU·PCIe 세대·동시성 또는 목표 업무에 직접 순위화할 수 없다. 기업 문서 검토 데이터와 Agentic AI 업무 지연은 검증되지 않았다.

판정: 미검수

- E-research_infinigen-3-1 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> We run the experiments on a system equipped with an NVIDIA RTX A6000 GPU [44] with 48GB of memory and an Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory. PCIe 3.0×16 interconnects the CPU and GPU.

- E-research_infinigen-3-2 | infinigen | 물리 페이지 11 | Table 2: Perplexity on WikiText-2 and PTB with 2048 sequence length with or without KV cache memory limits. Lower is better. Scheme OPT-6.7B OPT-13B OPT-30B Lla

> InfiniGen achieves 1.63×-32.93× speedups over the baselines. The performance benefit mainly comes from the significantly reduced amount of KV cache to load from the CPU memory due to our dynamic approach.

## research_infinigen-4

**기술:** InfiniGen · **주장 종류:** 팀 추론

InfiniGen의 공개 논문 근거상 잠정 TRL은 4 수준(보수적 범위 3~4)이며, 여러 LLM의 실험실 검증은 있으나 기업 업무 운용과 독립 재현까지는 실증되지 않았다.

**조건:** 판단 기준은 arXiv:2406.19707v1(2024-06-28)의 논문 구현과 RTX A6000 기반 실험, OPT·Llama-2 모델 크기 변화, batch·sequence length 변화, downstream accuracy·perplexity·latency·memory 분석이다. 이는 실험실 시스템 검증에 해당하는 근거로 보았으며, 대표 사용조건 또는 실제 지속운용 검증으로 확대하지 않았다. 공개 저장소가 source_metadata에 식별되지만, 제공 발췌만으로 독립 실행 성공이나 라이선스 적합성을 확인하지 않았다.

**한계:** TRL은 논문이 인증한 값이 아니라 공개 실험 범위에 대한 팀의 보수적 해석이다. 논문 구현은 대표 LLM·배치·sequence length 실험을 제공하지만, 기업 IT 사업 문서 검토 Agentic AI의 장문·반복·동시 요청, 운영환경 SLA, 장애복구, 보안, 독립 재현과 공식 라이선스·실행 절차는 제공 발췌에서 확인되지 않는다.

판정: 미검수

- E-research_infinigen-4-1 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> We implement InfiniGen on a modern offloading-based inference system and demonstrate that it greatly out- performs the existing KV cache management methods

- E-research_infinigen-4-2 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> We use Open Pre-trained Transformer (OPT) models [77] with 6.7B, 13B, and 30B parameters for evaluation. The 7B and 13B models of Llama- 2 [60] are also used

- E-research_infinigen-4-3 | infinigen | 물리 페이지 14 | 0 20 40 60 80 100 2K 16K 128K 1M Percentage (%) Layer 0 Layer 12 Layer 24 Layer 30 (a) Sequence Length Layer 18, Head 30 Attention Weight 0 8K 16K4K 12K (b) Ite

> InfiniGen exploits the attention input of the previous layer to speculatively prefetch the KV cache of important tokens.

## market-1

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI: 공개 논문·공식 코드와 LLM 벤치마크는 생태계 신호지만, 기업 IT 문서 검토 Agent의 공개 채택·상용 운영은 제공 범위에서 확인되지 않는다.

**조건:** KIVI 논문 2402.02750v2와 공식 저장소 README 버전 baa1095e6edf8263bbf20507f0d1ce444c3cb57d97d5f5677c2ac19c3b934bbf를 기준으로 판단했다. Llama·Falcon·Mistral, LM-Eval·LongBench·Needle-in-a-Haystack 평가와 공개 코드만 확인했으며, RFP·계약·운영 SLA·보안·장애복구·Agent 도구 연계는 미확인이다.

**한계:** 논문과 공식 저장소의 공개는 구현·재현성에 관한 신호이지 고객 도입이나 생산환경 운용 증거가 아니다. SK AX 또는 공개 AiPMO 업무에서 KIVI를 채택했다는 사실은 확인되지 않는다.

판정: 미검수

- E-market-1-1 | kivi | 물리 페이지 1 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Zirui Liu * 1 Jiayi Yuan* 1 Hongye Jin 2 Shaochen (Henry) Zhong 1 Zhaozhuo Xu 3 Vladimir Braverman

> With hardware-friendly implementation, KIVI can enable Llama, Falcon, and Mistral models to maintain almost the same quality while using 2.6× less peak memory

- E-market-1-2 | kivi | 물리 페이지 1 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Zirui Liu * 1 Jiayi Yuan* 1 Hongye Jin 2 Shaochen (Henry) Zhong 1 Zhaozhuo Xu 3 Vladimir Braverman

> The source code is available at https://github.com/jy-yuan/KIVI.

## market-2

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI: GPU KV 메모리와 배치 용량이 핵심 병목이면 AWQ·GPTQ나 시스템형 vLLM·S3보다 직접적인 선택지지만, MQA·GQA 모델은 4비트와 품질 회귀를 우선 검토해야 한다.

**조건:** KIVI 논문 2402.02750v2의 관련 연구, Falcon multi-query attention의 2비트·4비트 결과, 공식 LongBench 문서의 MQA·GQA 권고를 사용했다. KIVI 선택 조건은 KV 캐시가 병목인지, 모델이 MHA·MQA·GQA인지, 문서 검토 정확도와 인용 품질을 별도 평가할 수 있는지다. 비용·지연의 직접 비교 수치는 미확인이다.

**한계:** 대체 기술과 KIVI를 동일 모델·GPU·문서 길이·배치에서 직접 비교한 자료는 없다. Falcon 사례의 품질 저하가 기업 문서 검토의 사실성·인용 정확도 저하로 환산되는지는 미검증이다.

판정: 미검수

- E-market-2-1 | kivi | 물리 페이지 8 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 3: Performance comparison between 16bit, 4-bit per- token quantization, four fake 2bit KV ca

> A main branch of LLM quantization is weight-only quantization, which in- volves the quantization of model weights to lower precision. For instance, AWQ (Lin et al., 2023) cleverly quantizes model weights to INT4 and INT3 using an activation-aware manner.

- E-market-2-2 | kivi | 물리 페이지 9 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 4: Performance evaluation of KIVI on various models across a range of benchmarks in LongBenc

> vLLM (Kwon et al., 2023) and S3 (Jin et al., 2023) are system-level works, which include memory management

- E-market-2-3 | expanded_cdd9ac36bdc16180 | 물리 페이지 None | snapshot block 1, character 0

> For multiquery attention or group query attention, since the keys and values are already compressed, we recommend using KIVI-4

## market-3

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI: 금전적 도입·운영비는 산정할 수 없으며, Transformers 통합과 CUDA·Triton 커널 검증, 잔여 FP 캐시 관리 및 업무별 품질 회귀시험이 주요 부담으로 남는다.

**조건:** 도입 부담은 KIVI 논문 2402.02750v2의 Hugging Face 기반 구현, CUDA 역양자화·행렬곱 융합, Triton 커널, grouped/residual 캐시 구조에 근거한다. 공식 저장소의 제공 버전은 baa1095e6edf8263bbf20507f0d1ce444c3cb57d97d5f5677c2ac19c3b934bbf다. ShareGPT 단일 A100 실험은 비용 산정이 아닌 효율 참고치이며, 실제 인프라 가격과 반복 운영 인력은 미확인이다.

**한계:** 인력·GPU·소프트웨어 라이선스·유지보수·장애 대응의 가격과 운영비는 제공 자료에서 확인되지 않는다. 양자화 설정이 문서 검토 품질과 회귀시험 범위에 미치는 비용도 미검증이다.

판정: 미검수

- E-market-3-1 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> We provide a hardware-friendly imple- mentation for running KIVI on GPUs. To minimize the overhead, we have fused the dequantization process with matrix multiplication, e.g., Q_MatMul in Figure 3, using CUDA.

- E-market-3-2 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> We use the Hugging Face Transformers codebase and implement the KIVI algorithm upon it.

- E-market-3-3 | kivi | 물리 페이지 2 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache per-token quantization per-channel quantization Figure 1: Definition of per-token and per-channel 

> We only apply group-wise quanti- zation to the grouped key cache and value cache, while the residual key cache and value cache are kept in full precision.

## market-4

**기술:** InfiniGen · **주장 종류:** 팀 추론

InfiniGen: 공개 논문·저장소와 OPT·Llama-2 실험은 offloading 구현의 성숙도 신호지만, 기업 IT 문서 검토 Agent의 공개 채택·상용 운영은 제공 범위에서 확인되지 않는다.

**조건:** arXiv 2406.19707v1(2024-06-28)과 공식 저장소 README 버전 f6a08e32c16d3fdbe8839a95775f2b1e2a2690e36e6ee9d8ec683d6c24e89a90을 기준으로 했다. OPT 6.7B·13B·30B 및 Llama-2 7B·13B, offloading 기반 실험은 확인했지만 Agent의 도구 호출·문서 권한·감사로그·SLA·보안 운영은 미확인이다.

**한계:** 논문 구현과 공식 저장소 README는 기술 공개와 실험 재현성의 신호이지 실제 구매·배포·지속운영을 입증하지 않는다. 고객·매출·채택률·기업 문서 검토 레퍼런스는 확인되지 않는다.

판정: 미검수

- E-market-4-1 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> We implement InfiniGen on a modern offloading-based inference system and demonstrate that it greatly out- performs the existing KV cache management methods

- E-market-4-2 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> We use Open Pre-trained Transformer (OPT) models [77] with 6.7B, 13B, and 30B parameters for evaluation. The 7B and 13B models of Llama- 2 [60] are also used

- E-market-4-3 | infinigen_repo | 물리 페이지 None | snapshot block 8, character 0

> In this paper, we present InfiniGen, a novel KV cache management framework tailored for long-text generation

## market-6

**기술:** InfiniGen · **주장 종류:** 팀 추론

InfiniGen: 금전적 도입·운영비는 공개 자료로 산정할 수 없고, CPU KV pool·PCIe 전송 관리와 partial weight·alpha 설정의 메모리·품질 검증 부담을 추가한다.

**조건:** 도입 부담은 InfiniGen 2406.19707v1의 RTX A6000 48GB, Xeon Gold 6136, DDR4-2666 96GB, PCIe 3.0×16 구성과 OPT-6.7B 민감도 실험을 참고했다. alpha·partial weight ratio는 OPT-6.7B, 입력 1920·출력 128, batch 8, WinoGrande 조건에서 평가됐고, ratio 0.3은 정확도와 메모리 절충으로 선택됐다. 실제 구매비·유지보수비·장애 대응비는 미확인이다.

**한계:** 서버·CPU 메모리·PCIe·전력·운영 인력의 가격, 예측 오류율과 세대별 PCIe 민감도는 제공 자료에서 확인되지 않는다. partial weight ratio 0.3과 alpha 설정은 보편적 최적값이 아니다.

판정: 미검수

- E-market-6-1 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> We run the experiments on a system equipped with an NVIDIA RTX A6000 GPU [44] with 48GB of memory and an Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory. PCIe 3.0×16 interconnects the CPU and GPU.

- E-market-6-2 | infinigen | 물리 페이지 13 | 0 5 10 15 FlexGen INT4 H2O InfiniGen Ideal Latency (ms) Attention FFN Data Transfer Prediction 28.0 H2O Figure 18: Latency breakdown of a Transformer block for 

> increasing the partial weight ratio results in higher memory consumption for partial weights and key cache (e.g., doubling the ratio doubles the memory consumption overhead).

- E-market-6-3 | infinigen | 물리 페이지 12 | 0 2 4 6 512 1024 1536 2048 Speedup INT4 H2O InfiniGen 0 2 4 6 6.7B 13B 30B Speedup INT4 H2O InfiniGen (b) Model Size(a) Sequence Length H2O H2O Figure 16: Speed

> Increasing alpha results in fetching more KV entries to the GPU

## stakeholder-1

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI는 KV 캐시 양자화로 문서 검토 Agent의 동시 처리 여력을 높일 가능성이 있지만, RFP·계약서의 인용·요약 정확도는 별도 검증해야 해 사용자 확인 부담이 남는다.

**조건:** KIVI 논문 arXiv:2402.02750v2 Figure 5의 조건은 Llama-2-7B, 2비트 KIVI와 16비트 baseline, ShareGPT 평균 입력 161토큰·출력 338토큰, 단일 NVIDIA A100 80GB이다. 메모리 한계까지 배치를 늘려 최대 메모리와 throughput을 비교했으며, 기업의 장문·반복·동시 요청은 적용 가정이다.

**한계:** KIVI의 처리량 결과는 기업 문서 검토가 아닌 ShareGPT 기반 합성 서비스 워크로드에서 얻었으며, 문서 사실성·인용 정확도·Agent 도구 호출 품질은 확인되지 않았다.

판정: 미검수

- E-stakeholder-1-1 | kivi | 물리 페이지 8 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 3: Performance comparison between 16bit, 4-bit per- token quantization, four fake 2bit KV ca

> with similar maximum memory us- age, KIVI enables up to 4× larger batch size and gives 2.35×∼ 3.47× larger throughput.

- E-stakeholder-1-2 | aipmo | 물리 페이지 None | snapshot block 15, character 0

> AiPMO는 RFP, 계약서, 사업계획서, 발주 문서 등 다양한 사업 자료를 자동으로 분석해 검토가 필요한 핵심 항목을 식별합니다.

## stakeholder-2

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI는 Hugging Face·CUDA·Triton 기반 구현으로 운영 통합을 검토할 수 있으나, group size·residual length별 품질과 메모리·지연 회귀를 관측하는 부담은 운영자에게 남는다.

**조건:** KIVI v2는 Hugging Face Transformers 기반이며 CUDA로 역양자화와 행렬곱을 융합하고 Triton group-wise kernel을 구현했다. 실험 기본값은 group size 32, residual length 128이고, residual length 32·64·96·128 및 group size 변화가 품질에 미치는 영향을 별도 평가했다.

**한계:** 제공 발췌에서는 Agent 도구 연동, 다중 GPU, 운영 SLA, 장애복구 절차와 업무별 품질 회귀가 확인되지 않았다. 논문 설정을 운영 환경의 보편적 최적값으로 일반화할 수 없다.

판정: 미검수

- E-stakeholder-2-1 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> We use the Hugging Face Transformers codebase and implement the KIVI algorithm upon it.

- E-stakeholder-2-2 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> To minimize the overhead, we have fused the dequantization process with matrix multiplication, e.g., Q_MatMul in Figure 3, using CUDA.

- E-stakeholder-2-3 | kivi | 물리 페이지 7 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache 0.5K2K4K7K9K11K13K16K18K20K Word Count 0.0 0.11 0.22 0.33 0.44 0.56 0.67 0.78 0.89 1.0 Depth 20K w

> The effect of residual length. We fix the group size at 32 and vary the residual length across 32, 64, 96, and 128.

## stakeholder-3

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI는 GPU 메모리 절감과 공개 구현으로 인프라 부담을 낮출 여지가 있지만, 모델 구조별 2비트 손실 위험과 라이선스·보안·결과 책임 기준 확인이 구매 승인 조건이다.

**조건:** KIVI arXiv:2402.02750v2 Table 3은 Falcon-7B의 16비트 baseline과 KIVI-2·KIVI-4를 비교하며, Falcon의 multi-query attention 구조에서는 2비트보다 4비트가 정확도 유지에 필요하다고 보고했다. Table 5는 Llama2-13B·GSM8K에서 group size 32·64·128을 비교했고 128에서 17.29를 기록했다. 공개 저장소는 식별되지만 라이선스 적합성은 별도 확인 대상이다.

**한계:** Falcon·GSM8K 등 논문 벤치마크의 손실을 기업 문서 검토 위험으로 직접 환산할 수 없다. 제공 자료에서 공식 라이선스, 보안 통제와 생성 결과 책임분담은 확인되지 않았다.

판정: 미검수

- E-stakeholder-3-1 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> 4bit KIVI is needed to maintain the accuracy, while 2bit KIVI may have a large accuracy drop in this case.

- E-stakeholder-3-2 | kivi | 물리 페이지 1 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Zirui Liu * 1 Jiayi Yuan* 1 Hongye Jin 2 Shaochen (Henry) Zhong 1 Zhaozhuo Xu 3 Vladimir Braverman

> The source code is available at https://github.com/jy-yuan/KIVI.

- E-stakeholder-3-3 | kivi | 물리 페이지 9 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 4: Performance evaluation of KIVI on various models across a range of benchmarks in LongBenc

> Llama2-13B 32 20.77 64 21.00 128 17.29

## stakeholder-4

**기술:** InfiniGen · **주장 종류:** 적용 가정

InfiniGen은 CPU KV pool에서 필요한 토큰만 GPU로 가져와 장문·반복 요청의 지연 완화를 기대하게 하지만, 예측 누락이 문서 인용·근거 품질에 미치는 영향은 사용자가 검증해야 한다.

**조건:** InfiniGen arXiv:2406.19707v1은 prefill에서 partial weight를 만들고, 디코딩 중 이전 레이어 입력으로 다음 레이어 attention pattern을 추정한 뒤 alpha 기준으로 KV를 선택한다. 목표 업무에서는 원문 근거 회수율, 인용 정확도, 요약 품질과 누락 시 사용자 검토 절차를 별도 평가해야 한다.

**한계:** 논문은 일반 LLM 추론에서 출력 품질 유지를 보고했지만, RFP·계약 문서의 근거 회수율·인용 누락률·반복 질의 일관성은 확인되지 않았다. attention prediction 오류율도 제공 발췌에서 확인되지 않았다.

판정: 미검수

- E-stakeholder-4-1 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> By dynamically adjusting the number of KV entries to prefetch, InfiniGen brings only the necessary amount of the KV cache to the GPU, thereby greatly reducing the overhead of the KV cache transfer.

- E-stakeholder-4-2 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> InfiniGen only prefetches the keys and values of the tokens with an attention score larger than the highest attention score minus alpha.

- E-stakeholder-4-3 | aipmo | 물리 페이지 None | snapshot block 15, character 0

> 문서 내 표현 방식이나 형식이 달라서 AI Agent가 구조를 파악하고, 일정, 산출물 항목을 도출합니다.

## stakeholder-5

**기술:** InfiniGen · **주장 종류:** 팀 추론

InfiniGen은 CPU 메모리와 GPU 사이의 동적 prefetch로 전송 병목을 줄일 수 있지만, alpha·partial weight ratio·PCIe 상태와 CPU·GPU 자원을 함께 관측하는 운영 부담이 커진다.

**조건:** InfiniGen v1 Figure 17의 민감도 실험은 OPT-6.7B, 입력 1920·출력 128, batch 8, WinoGrande에서 alpha와 partial weight ratio를 평가했다. 논문은 ratio 0.3을 선택했고, 시스템 실험은 RTX A6000 48GB, Xeon Gold 6136, DDR4-2666 96GB, PCIe 3.0×16에서 수행했다. 운영 시 전송량, GPU·CPU 메모리, alpha, ratio와 지연을 함께 관측해야 한다.

**한계:** 제공 발췌에는 prediction 오류율, PCIe 세대별 민감도, 장시간 반복 요청 안정성, 운영 장애복구 절차가 없다. ratio 0.3은 논문 실험의 절충이지 보편적 최적값이 아니다.

판정: 미검수

- E-stakeholder-5-1 | infinigen | 물리 페이지 13 | 0 5 10 15 FlexGen INT4 H2O InfiniGen Ideal Latency (ms) Attention FFN Data Transfer Prediction 28.0 H2O Figure 18: Latency breakdown of a Transformer block for 

> increasing the partial weight ratio results in higher memory consumption for partial weights and key cache

- E-stakeholder-5-2 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> PCIe 3.0×16 interconnects the CPU and GPU.

- E-stakeholder-5-3 | infinigen | 물리 페이지 12 | 0 2 4 6 512 1024 1536 2048 Speedup INT4 H2O InfiniGen 0 2 4 6 6.7B 13B 30B Speedup INT4 H2O InfiniGen (b) Model Size(a) Sequence Length H2O H2O Figure 16: Speed

> Increasing alpha results in fetching more KV entries to the GPU

## stakeholder-6

**기술:** InfiniGen · **주장 종류:** 팀 추론

InfiniGen은 GPU 증설 대신 CPU 메모리·PCIe 구성을 활용하는 선택지지만, 특정 하드웨어 의존성과 CPU 내 KV 보관의 보안·라이선스·복구 기준 확인이 구매 승인의 전제다.

**조건:** InfiniGen arXiv:2406.19707v1 Figure 14는 OPT-13B, sequence length 2048(입력 1920·출력 128), batch 20, RTX A6000 48GB, Xeon Gold 6136, DDR4-2666 96GB, PCIe 3.0×16에서 prefill·decoding latency를 FlexGen, UVM, H2O(KV budget 20%), FlexGen+INT4와 비교했다. 보고된 1.63×–32.93×는 inference latency 기준의 시스템 실행 결과이며 모델 정밀도와 해당 latency 실험의 데이터셋은 미확인이다. 구매 시 하드웨어 호환성, CPU 메모리 보호, 라이선스와 복구 책임을 별도 확인해야 한다.

**한계:** 수치는 논문이 제시한 특정 실험실 시스템의 결과이며 기업 구매 효과로 확정할 수 없다. 제공 발췌에서 CPU KV 잔존·보호 통제, 공식 라이선스, 독립 재현 절차와 장애복구 기준은 확인되지 않았다.

판정: 미검수

- E-stakeholder-6-1 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> We run the experiments on a system equipped with an NVIDIA RTX A6000 GPU [44] with 48GB of memory and an Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory. PCIe 3.0×16 interconnects the CPU and GPU.

- E-stakeholder-6-2 | infinigen | 물리 페이지 11 | Table 2: Perplexity on WikiText-2 and PTB with 2048 sequence length with or without KV cache memory limits. Lower is better. Scheme OPT-6.7B OPT-13B OPT-30B Lla

> InfiniGen achieves 1.63×-32.93× speedups over the baselines.

- E-stakeholder-6-3 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> We implement InfiniGen on a modern offloading-based inference system and demonstrate that it greatly out- performs the existing KV cache management methods

## domain-1

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI는 KV 캐시를 2비트로 줄이고 잔여 구간은 FP로 유지하므로, 장문·반복 문서 검토 Agent의 동시 처리 후보가 되지만 업무 품질 적합성은 간접 근거다.

**조건:** KIVI 논문 v2의 Llama-2-7B, 2비트 KIVI와 FP16 baseline, ShareGPT 기반 합성 서비스 워크로드, 평균 입력 161토큰·출력 338토큰, 단일 NVIDIA A100 80GB 조건을 적용 가능성의 간접 근거로 사용했다. 배치는 메모리 한계까지 증가시켰고 지표는 최대 메모리와 throughput이다. 목표 업무는 RFP·계약서·사업계획서·발주 문서의 핵심 항목 식별 흐름으로 가정한다.

**한계:** AiPMO의 문서 분석 설명과 KIVI 실험은 결합 검증이 아니다. 조항 추출, 인용 보존, 요약 사실성, RAG·도구 호출 품질은 확인되지 않았다.

판정: 미검수

- E-domain-1-1 | kivi | 물리 페이지 2 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache per-token quantization per-channel quantization Figure 1: Definition of per-token and per-channel 

> KIVI quantizes key cache per-channel and quantizes value cache per-token.

- E-domain-1-2 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> KIVI maintains a full precision KV cache sliding window for the local relevant tokens.

- E-domain-1-3 | kivi | 물리 페이지 8 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 3: Performance comparison between 16bit, 4-bit per- token quantization, four fake 2bit KV ca

> with similar maximum memory us- age, KIVI enables up to 4× larger batch size and gives 2.35×∼ 3.47× larger throughput.

- E-domain-1-4 | aipmo | 물리 페이지 None | snapshot block 15, character 0

> AiPMO는 RFP, 계약서, 사업계획서, 발주 문서 등 다양한 사업 자료를 자동으로 분석해 검토가 필요한 핵심 항목을 식별합니다.

## domain-2

**기술:** KIVI · **주장 종류:** 저자 보고 결과

KIVI는 2비트 압축 이득이 모델 구조와 설정에 좌우된다. Falcon의 이미 압축된 KV에서는 4비트가 필요할 수 있고, 큰 group size는 대표 태스크 정확도를 낮출 위험이 있다.

**조건:** KIVI v2 Table 3은 Falcon의 multi-query attention 구조에서 16비트, 4비트, 2비트 설정을 비교했으며 세부 데이터셋·전체 지표는 발췌상 미확인이다. Table 5는 Llama2-13B, GSM8K exact-match accuracy, group size 32·64·128 조건으로 각각 20.77·21.00·17.29를 제시했다. 문서 적용에서는 2·4비트, group size, residual length별 조항 recall과 인용 precision을 비교해야 한다.

**한계:** 확인된 손실은 Falcon 및 GSM8K 등 논문 벤치마크 결과다. 기업 문서의 조항 누락·근거 인용 오류·장시간 반복 요청 오류율은 제공 발췌에서 확인되지 않았다.

판정: 미검수

- E-domain-2-1 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> 4bit KIVI is needed to maintain the accuracy, while 2bit KIVI may have a large accuracy drop in this case.

- E-domain-2-2 | kivi | 물리 페이지 7 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache 0.5K2K4K7K9K11K13K16K18K20K Word Count 0.0 0.11 0.22 0.33 0.44 0.56 0.67 0.78 0.89 1.0 Depth 20K w

> the performance significantly decreases when the group size reaches 128.

- E-domain-2-3 | kivi | 물리 페이지 9 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 4: Performance evaluation of KIVI on various models across a range of benchmarks in LongBenc

> Llama2-13B 32 20.77 64 21.00 128 17.29

## domain-3

**기술:** KIVI · **주장 종류:** 팀 추론

KIVI의 논문 평가는 일반·장문 생성과 메모리·처리량을 다루지만, 기업 문서 Agent의 정확도와 p95 지연을 검증하지 않아 업무 도입 판정에는 별도 시험이 필요하다.

**조건:** KIVI v2의 LM-Eval은 CoQA exact match, TruthfulQA BLEU, GSM8K exact match를 사용하고 LongBench·NIAH로 장문 처리를 평가한다. 효율 실험은 Llama-2-7B, ShareGPT 기반 평균 입력 161·출력 338토큰, 2비트 KIVI residual length 32·128, FP16 baseline, 단일 A100 80GB이며 throughput·최대 메모리를 측정했다. 목표 업무에서는 동일 문서 세트로 조항·요구사항 추출, 인용 precision/recall, 요약 사실성, 도구 호출 성공률, p95 지연, 최대 동시 요청과 GPU 메모리를 추가 측정해야 한다.

**한계:** 논문 벤치마크와 NIAH는 문서 검토의 요구사항 추출·인용·리스크 판정 검증이 아니다. 실제 기업 Agent의 데이터 분포, 동시성, 운영 안정성은 미확인이다.

판정: 미검수

- E-domain-3-1 | kivi | 물리 페이지 6 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache value tensors are added to XKr and XVr in full precision, KIVI maintains a full precision KV cache

> For LM-eval, we adopt CoQA (Exact match accuracy), TruthfulQA (BLEU score), and GSM8K (Exact match accuracy).

- E-domain-3-2 | kivi | 물리 페이지 13 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache B. NIAH Setting We largely follows the passkey retrieval prompt template of Mohtashami and Jaggi (

> We also show result of Needle-in-a-Haystack Test in Figure 4.

- E-domain-3-3 | kivi | 물리 페이지 8 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache Table 3: Performance comparison between 16bit, 4-bit per- token quantization, four fake 2bit KV ca

> Figure 5: Memory usage and throughput comparison be- tween 2bit KIVI and 16bit baseline.

- E-domain-3-4 | aipmo | 물리 페이지 None | snapshot block 17, character 0

> 반복적이고 정형화된 검토는 AI Agent가 수행하고, 최종 판단은 전문가가 보완하는 Human-in-the-loop 구조로 신뢰성을 확보합니다.

## domain-4

**기술:** InfiniGen · **주장 종류:** 적용 가정

InfiniGen은 CPU KV pool에 캐시를 두고 다음 레이어에 필요한 항목만 GPU로 prefetch하므로 장문·반복 요청에 맞을 가능성이 있지만, CPU 메모리와 PCIe 계층을 전제로 한다.

**조건:** InfiniGen v1의 offloading-based inference, CPU KV pool, Layer i−1 입력과 partial weight를 이용한 다음 레이어 attention 추정, alpha 기반 선택적 prefetch를 전제로 한다. 적용 후보 환경에서는 GPU·CPU 메모리 용량, PCIe 세대·대역폭, KV 전송량, 문서 길이, 반복 질의율과 동시 요청을 함께 측정해야 한다.

**한계:** 문서 검토에 대한 적용은 대용량 자료와 반복 질의를 전제로 한 시나리오다. 목표 업무에서 CPU offload가 실제로 유리한지, Agent orchestration·보안 격리가 가능한지는 확인되지 않았다.

판정: 미검수

- E-domain-4-1 | infinigen | 물리 페이지 2 | skewing the Transformer architecture query and key matrices to emphasize certain important columns. During the prefill stage, while the prompt and input of an i

> By dynamically adjusting the number of KV entries to prefetch, InfiniGen brings only the necessary amount of the KV cache to the GPU, thereby greatly reducing the overhead of the KV cache transfer.

- E-domain-4-2 | infinigen | 물리 페이지 6 | number of key tokens required to reach 0.9 varies even for the adjacent query tokens; for instance, the 998th, 999th, 1000th, 1001st, and 1002nd tokens need 172

> InfiniGen, which enables offloading the KV cache with low data transfer overhead.

- E-domain-4-3 | aipmo | 물리 페이지 None | snapshot block 15, character 0

> 이를 통해 초기 검토 단계에서 고객 요구사항에 빠르게 대앙할 수 있으며, 대용량 문서도 효율적으로 처리 가능합니다

## domain-5

**기술:** InfiniGen · **주장 종류:** 팀 추론

논문 결과: InfiniGen은 필요한 KV를 예측해 전송량을 줄이며, partial weight ratio를 높이면 partial weights와 key cache의 메모리 사용량이 증가한다. 팀 추론: 예측 누락이 기업 문서의 근거 보존과 응답 지연에 미치는 영향은 업무 데이터로 확인해야 한다.

**조건:** 민감도 실험은 OPT-6.7B, 입력 1920·출력 128토큰, batch 8, WinoGrande accuracy 조건이다. alpha가 커지면 더 많은 KV를 fetch하고, partial weight ratio 증가는 partial weights와 key cache overhead를 증가시킨다. 목표 업무에서는 alpha·ratio별 attention miss rate, 근거 토큰 누락률, 조항 recall, p95/p99 지연, CPU·GPU 메모리와 PCIe 전송량을 측정해야 한다.

**한계:** 발췌에는 attention prediction 오류율, PCIe 구성별 민감도, 장시간 반복 요청 안정성 및 문서 근거 누락률이 없다. partial weight ratio 0.3은 논문 실험의 선택값이지 업무 최적값으로 확인된 것이 아니다.

판정: 미검수

- E-domain-5-1 | infinigen | 물리 페이지 5 | the KV cache size through key/value evictions at runtime within a constrained KV cache budget [37, 78]. However, all the prior works assume the persistence of a

> However, all the prior works assume the persistence of attention patterns across iterations

- E-domain-5-2 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> By reducing the amount of KV cache to load and compute, InfiniGen effectively reduces the loading latency

- E-domain-5-3 | infinigen | 물리 페이지 13 | 0 5 10 15 FlexGen INT4 H2O InfiniGen Ideal Latency (ms) Attention FFN Data Transfer Prediction 28.0 H2O Figure 18: Latency breakdown of a Transformer block for 

> increasing the partial weight ratio results in higher memory consumption for partial weights and key cache

## domain-6

**기술:** InfiniGen · **주장 종류:** 팀 추론

InfiniGen은 실제 GPU·CPU·PCIe 시스템에서 장문 batch 추론 지연을 측정했지만, 그 speedup은 기업 문서 Agent의 품질·동시성 성과로 전환되지 않아 별도 검증이 필요하다.

**조건:** InfiniGen v1 Figure 14는 OPT-13B, sequence length 2048(입력 1920·출력 128), batch 20, RTX A6000 48GB, Xeon Gold 6136, DDR4-2666 96GB, PCIe 3.0×16에서 prefill·decoding latency를 측정하고 UVM, H2O, FlexGen, FlexGen+INT4와 비교해 1.63×–32.93× speedup을 보고했다. 정확도 실험은 WinoGrande 등 lm-evaluation-harness와 WikiText-2·PTB를 사용했다. 문서 적용에서는 요구사항·조항 추출, 인용 보존율, 리스크 precision/recall, 요약 사실성, p95 지연, 전송량과 동시성 확장성을 full-KV baseline과 비교해야 한다.

**한계:** 이는 논문에 제시된 시스템 실행 결과이지 기업 문서 업무 결과가 아니다. KIVI와 동일 모델·GPU·컨텍스트·배치에서 직접 비교한 실험은 확인되지 않았고, 반복 측정 횟수와 소프트웨어 세부도 미확인이다.

판정: 미검수

- E-domain-6-1 | infinigen | 물리 페이지 9 | token, it does not noticeably hurt the accuracy of the model since it accounts for less than 1% of importance (≈ 1/148.4) after softmax. Thus, InfiniGen only pr

> We run the experiments on a system equipped with an NVIDIA RTX A6000 GPU [44] with 48GB of memory and an Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory. PCIe 3.0×16 interconnects the CPU and GPU.

- E-domain-6-2 | infinigen | 물리 페이지 11 | Table 2: Perplexity on WikiText-2 and PTB with 2048 sequence length with or without KV cache memory limits. Lower is better. Scheme OPT-6.7B OPT-13B OPT-30B Lla

> InfiniGen achieves 1.63×-32.93× speedups over the baselines.

- E-domain-6-3 | infinigen | 물리 페이지 12 | 0 2 4 6 512 1024 1536 2048 Speedup INT4 H2O InfiniGen 0 2 4 6 6.7B 13B 30B Speedup INT4 H2O InfiniGen (b) Model Size(a) Sequence Length H2O H2O Figure 16: Speed

> The accuracy is evaluated with the WinoGrande task in lm-evaluation-harness.
