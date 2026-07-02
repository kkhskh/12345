# Top 100 Model Atlas V0

## 0. Status and Caveat

This is a seed atlas, not a final ranking. It is grouped by provider/model family and should not be cited as a leaderboard claim.

Values should be source-verified before publication. Closed models are external-only unless special research access exists. Open models may be mechanistically testable depending on hardware, license, tokenizer/logit access, activation access, and patching feasibility.

The seed rows below are grounded only as an atlas seed from `docs/notes/public_model_family_atlas_seed_2026_07_02.md` unless otherwise marked. Rows marked `TODO_VERIFY` require provider docs, model cards, leaderboard entries, or release papers before publication.

## 1. Source Categories

- LMArena / Chatbot Arena
- Artificial Analysis
- OpenRouter
- Stanford HELM
- Stanford FMTI
- Provider docs
- Hugging Face model cards
- model reports / release papers

## 2. Access Categories

- closed API / UI: externally callable or usable, with hidden weights and internals.
- closed system wrapper: product or agent wrapper over one or more hidden models/tools.
- open-weight API: weights are public or described as open-weight, but current use may be hosted.
- open-weight local: weights are available for local analysis subject to hardware and license.
- research-control suite: open checkpoints or controls make mechanistic research unusually feasible.
- media API: image/video/audio generation or editing API with hidden internals.
- tool/RAG/agent system: composite system where model decisions must be separated from scaffold/tool/retrieval decisions.

## 3. Mechanistic Access Categories

- `E`: external-only.
- `P`: partial mechanistic access.
- `M`: internal mechanistic possible.
- `M*`: possible but hardware-heavy.
- `M++`: unusually good for mechanistic research.
- `U`: unknown.

## 4. Seed Table

| index | canonical model/system | developer/provider | access category | modality | family | public architecture/params status | context status | source notes | mechanistic access | atlas role | TODO verification status |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GPT-5.6 preview | OpenAI | closed API / UI | text+vision+tools | closed frontier API | not public | not fully public | seed: OpenAI docs | E | closed frontier stratum | TODO_VERIFY |
| 2 | GPT-5.5 | OpenAI | closed API / UI | text+vision+tools | closed frontier API | not public | seed says 1M | seed: OpenAI, AA | E | closed frontier stratum | TODO_VERIFY |
| 3 | GPT-5.5 Pro | OpenAI | closed API / UI | reasoning/text+vision | closed frontier API | not public | public priced tier | seed: OpenAI | E | reasoning API stratum | TODO_VERIFY |
| 4 | GPT-5.4 | OpenAI | closed API / UI | text+vision+tools | closed frontier API | not public | seed says 1M | seed: OpenAI | E | closed frontier stratum | TODO_VERIFY |
| 5 | GPT-5.4 Pro | OpenAI | closed API / UI | reasoning | closed frontier API | not public | public priced tier | seed: OpenAI | E | reasoning API stratum | TODO_VERIFY |
| 6 | GPT-5.4 mini | OpenAI | closed API / UI | text+vision+tools | closed frontier API | not public | seed says 400K | seed: OpenAI | E | cost/speed stratum | TODO_VERIFY |
| 7 | GPT-5.4 nano | OpenAI | closed API / UI | low-cost text/tool | closed frontier API | not public | not fully reviewed | seed: OpenAI | E | cost/speed stratum | TODO_VERIFY |
| 8 | GPT Image 2 | OpenAI | media API | image generation/editing | diffusion/media or media system | not public | n/a | seed: OpenAI pricing | E | generative media basin | TODO_VERIFY |
| 9 | GPT-Realtime-2 | OpenAI | closed API / UI | speech/audio realtime | audio/realtime system | not public | n/a | seed: OpenAI docs | E | multimodal/audio stratum | TODO_VERIFY |
| 10 | Sora 2 Pro | OpenAI | media API | video generation | diffusion/media or media system | not public | n/a | seed: OpenAI pricing | E | generative media basin | TODO_VERIFY |
| 11 | Claude Fable 5 | Anthropic | closed API / UI | text+vision+tools | closed frontier API | not public | seed says 1M | seed: Anthropic, LMArena, AA | E | closed frontier stratum | TODO_VERIFY |
| 12 | Claude Mythos 5 | Anthropic | closed API limited | cyber/agent workflows | closed frontier API | not public | same class as Fable docs | seed: Anthropic docs | E | specialized agent stratum | TODO_VERIFY |
| 13 | Claude Opus 4.8 | Anthropic | closed API / UI | text+vision+agentic coding | closed frontier API | not public | seed says 1M | seed: Anthropic, AA | E | coding/agent stratum | TODO_VERIFY |
| 14 | Claude Opus 4.7 | Anthropic | closed API / UI | text+vision+reasoning | closed frontier API | not public | public | seed: AA | E | reasoning API stratum | TODO_VERIFY |
| 15 | Claude Sonnet 5 | Anthropic | closed API / UI | text+vision+coding | closed frontier API | not public | seed says 1M | seed: Anthropic docs | E | coding/agent stratum | TODO_VERIFY |
| 16 | Claude Haiku 4.5 | Anthropic | closed API / UI | fast text+vision | closed frontier API | not public | seed says 200K | seed: Anthropic docs | E | cost/speed stratum | TODO_VERIFY |
| 17 | Claude Code / Claude agentic tooling | Anthropic | closed system wrapper | coding agent/tool system | tool/RAG/agent system | system wrapper | depends on model | seed: Anthropic docs | E | scaffold/tool stratum | TODO_VERIFY |
| 18 | Gemini 3.5 Flash | Google | closed API / UI | text+image+video+audio/PDF input | multimodal closed API | not public | seed says 1,048,576 input | seed: Gemini docs, AA | E | multimodal frontier stratum | TODO_VERIFY |
| 19 | Gemini 3.1 Pro Preview | Google | closed API preview | multimodal reasoning/coding | multimodal closed API | not public | seed says 1,048,576 input | seed: Gemini docs | E | multimodal frontier stratum | TODO_VERIFY |
| 20 | Gemini 3 Flash Preview | Google | closed API preview | multimodal | multimodal closed API | not public | public priced | seed: Gemini pricing | E | multimodal API stratum | TODO_VERIFY |
| 21 | Gemini 3.1 Flash-Lite | Google | closed API / UI | low-cost multimodal | multimodal closed API | not public | public docs | seed: Gemini docs | E | cost/speed stratum | TODO_VERIFY |
| 22 | Gemini Deep Research Max | Google | closed system wrapper | research agent/RAG | tool/RAG/agent system | system wrapper | n/a | seed: Gemini docs | E | RAG/agent stratum | TODO_VERIFY |
| 23 | Gemini Computer Use Preview | Google | closed system wrapper | computer-use agent | tool/RAG/agent system | not public | n/a | seed: Gemini docs | E | agent stratum | TODO_VERIFY |
| 24 | Gemini Antigravity Agent | Google | closed system wrapper | coding/agent system | tool/RAG/agent system | not public | n/a | seed: Gemini docs | E | coding/agent stratum | TODO_VERIFY |
| 25 | Nano Banana Pro / Gemini image | Google | media API | image generation | media system | not public | n/a | seed: Gemini docs | E | generative media basin | TODO_VERIFY |
| 26 | Veo 3.1 Preview | Google | media API | video generation | media system | not public | n/a | seed: Gemini docs | E | generative media basin | TODO_VERIFY |
| 27 | Gemini Robotics-ER 1.6 Preview | Google | closed API preview | robotics embodied reasoning | RL/world-model or robotics system | not public | n/a | seed: Gemini docs | E/U | robotics/control stratum | TODO_VERIFY |
| 28 | Grok 4.3 | xAI | closed API / UI | text+tools | closed frontier API | not public | seed says 1M | seed: xAI docs | E | closed frontier stratum | TODO_VERIFY |
| 29 | Grok Build 0.1 | xAI | closed API / UI | agentic coding | tool/RAG/agent system | not public | seed says 256K | seed: xAI docs | E | coding/agent stratum | TODO_VERIFY |
| 30 | Grok 4.1 Fast / Grok long-context variants | xAI | closed API / UI | text+tools | closed frontier API | not public | public via docs/leaderboards | seed: xAI, AA | E | cost/speed stratum | TODO_VERIFY |
| 31 | Perplexity Sonar | Perplexity | closed API / system | web-grounded QA | RAG/system | not public | not architecture-public | seed: Perplexity docs | E | RAG stratum | TODO_VERIFY |
| 32 | Perplexity Sonar Pro | Perplexity | closed API / system | web-grounded QA | RAG/system | not public | not architecture-public | seed: Perplexity docs | E | RAG stratum | TODO_VERIFY |
| 33 | Perplexity Sonar Reasoning Pro | Perplexity | closed API / system | reasoning+search | RAG/system | not public | not architecture-public | seed: Perplexity docs | E | RAG/reasoning stratum | TODO_VERIFY |
| 34 | Perplexity Sonar Deep Research | Perplexity | closed API / system | deep research+search | RAG/system | not public | not architecture-public | seed: Perplexity docs | E | RAG/agent stratum | TODO_VERIFY |
| 35 | Perplexity Agent API | Perplexity | tool/RAG/agent system | tool-using web agent | RAG/agent system | wrapper over models/tools | bounded tool loop | seed: Perplexity docs | E/P | scaffold/tool stratum | TODO_VERIFY |
| 36 | GLM-5.2 Max | Z.ai | open-weight API or API | long-horizon coding/agent | open/API long-context | source row needs verification | seed says 1M | seed: Z.ai, AA | M/P | open/API frontier stratum | TODO_VERIFY |
| 37 | GLM-5.2 | Z.ai | open-weight API | long-context agentic | open/API long-context | public docs | seed says 1M | seed: Z.ai docs | M/P | open/API frontier stratum | TODO_VERIFY |
| 38 | GLM-5.1 | Z.ai | API/possibly open-weight | long-horizon agentic | open/API long-context | public docs | not fully reviewed | seed: Z.ai docs | P/U | uncertain access stratum | TODO_VERIFY |
| 39 | GLM-5 | Z.ai | open-weight local+API | text/coding agentic | open dense/MoE TODO_VERIFY | public weights; MIT noted | public docs | seed: Z.ai docs | M | open mechanistic stratum | TODO_VERIFY |
| 40 | GLM-4.7 | Z.ai | open-weight API | coding/reasoning | open/API model | public docs | not fully reviewed | seed: Z.ai docs | M/P | open/API stratum | TODO_VERIFY |
| 41 | DeepSeek V4 Pro | DeepSeek | open-weight API | long-context reasoning/coding | MoE | seed says 1.6T total / 49B active public | seed says 1M | seed: DeepSeek, HF, Reuters | M* | open frontier MoE basin | TODO_VERIFY |
| 42 | DeepSeek V4 Flash | DeepSeek | open-weight API | efficient long-context | MoE | seed says 284B total / 13B active public | 1M class | seed: DeepSeek, NVIDIA/HF | M* | open frontier MoE basin | TODO_VERIFY |
| 43 | DeepSeek V3.2 | DeepSeek | open-weight API | reasoning/agents | open model | public open model | public | seed: DeepSeek docs | M* | open frontier stratum | TODO_VERIFY |
| 44 | DeepSeek V3.2-Speciale | DeepSeek | open-weight API | reasoning-first | open model | public open model | public | seed: DeepSeek docs | M* | reasoning stratum | TODO_VERIFY |
| 45 | DeepSeek V3.1 Terminus | DeepSeek | open-weight API | text/reasoning | open model | public open model | public | seed: provider/HF | M* | open frontier stratum | TODO_VERIFY |
| 46 | DeepSeek V3 | DeepSeek | open-weight local | text/coding | MoE | seed says 671B total / 37B active | public | seed: HF | M* | open frontier MoE basin | TODO_VERIFY |
| 47 | DeepSeek R1 | DeepSeek | open-weight local+API | reasoning | reasoning model | public reasoning model | public | seed: HF | M* | open reasoning stratum | TODO_VERIFY |
| 48 | DeepSeek R1-0528 | DeepSeek | open-weight local+API | reasoning update | reasoning model | public | public | seed: HF/provider | M* | open reasoning stratum | TODO_VERIFY |
| 49 | DeepSeek-R1-Distill-Qwen-32B | DeepSeek/Qwen | open-weight local | reasoning distill | decoder-only | dense Qwen-derived | public | seed: HF | M | research/open stratum | TODO_VERIFY |
| 50 | DeepSeek-R1-Distill-Llama-70B | DeepSeek/Meta | open-weight local | reasoning distill | decoder-only | Llama-derived | public | seed: HF | M | research/open stratum | TODO_VERIFY |
| 51 | Qwen3.6-35B-A3B | Alibaba/Qwen | open-weight local | text/reasoning | MoE | seed says 35B total / 3B active | public card | seed: HF | M | open MoE stratum | TODO_VERIFY |
| 52 | Qwen3.5-35B-A3B | Alibaba/Qwen | open-weight local | text/reasoning | MoE | seed says 35B total / 3B active | public card | seed: HF | M | open MoE stratum | TODO_VERIFY |
| 53 | Qwen3.5-Plus | Alibaba/Qwen | closed API | agentic/text/tools | API model | not fully public | seed says 1M default | seed: Qwen blog/Reuters | E/P | closed/API stratum | TODO_VERIFY |
| 54 | Qwen3-Max | Alibaba/Qwen | closed API | large agentic/coding | API model | seed says >1T public report | not fully reviewed | seed: Reuters/Qwen | E/P | closed/API stratum | TODO_VERIFY |
| 55 | Qwen3.5-Omni | Alibaba/Qwen | API/possibly open-weight variants | audio/vision/text | multimodal/MoE TODO_VERIFY | source row public in report | seed says 256K | seed: Qwen report | P/M if weights | multimodal stratum | TODO_VERIFY |
| 56 | Qwen3-235B-A22B | Alibaba/Qwen | open-weight local | text/reasoning | MoE | seed says 235B / 22B active | public | seed: Qwen | M* | open frontier MoE basin | TODO_VERIFY |
| 57 | Qwen3-30B-A3B | Alibaba/Qwen | open-weight local | text/reasoning | MoE | seed says 30B / 3B active | public | seed: Qwen | M | open MoE stratum | TODO_VERIFY |
| 58 | Qwen3-32B | Alibaba/Qwen | open-weight local | text/reasoning | decoder-only | dense 32B | public | seed: HF/Qwen | M | open dense stratum | TODO_VERIFY |
| 59 | Qwen3-14B | Alibaba/Qwen | open-weight local | text/reasoning | decoder-only | dense 14B | public | seed: HF/Qwen | M | open dense stratum | TODO_VERIFY |
| 60 | Qwen3-8B | Alibaba/Qwen | open-weight local | text/reasoning | decoder-only | dense 8B | public | seed: HF | M | open dense stratum | TODO_VERIFY |
| 61 | Qwen2.5-VL-72B | Alibaba/Qwen | open-weight local | vision-language | multimodal transformer | public VLM | public | seed: HF/Qwen | M* | multimodal open basin | TODO_VERIFY |
| 62 | Qwen2.5-Coder-32B | Alibaba/Qwen | open-weight local | code | decoder-only | decoder LLM | public | seed: HF/Qwen | M | coding/open stratum | TODO_VERIFY |
| 63 | QwQ-32B | Alibaba/Qwen | open-weight local | reasoning | decoder-only | decoder LLM | public | seed: HF/Qwen | M | reasoning/open stratum | TODO_VERIFY |
| 64 | Qwen-Scope SAE artifacts | Alibaba/Qwen researchers | research artifact | interpretability | SAE feature-space | SAEs for Qwen variants | n/a | seed: Qwen-Scope paper | M++ | SAE feature-space stratum | TODO_VERIFY |
| 65 | Llama 4 Scout | Meta | open-weight local | native multimodal | MoE/multimodal | seed says 17B active / 16 experts public | very long, reported 10M | seed: Meta/HF | M* | multimodal open basin | TODO_VERIFY |
| 66 | Llama 4 Maverick | Meta | open-weight local | native multimodal | MoE/multimodal | seed says 17B active / 128 experts public | public | seed: Meta/HF | M* | multimodal open basin | TODO_VERIFY |
| 67 | Llama 4 Behemoth preview | Meta | preview/not generally open | teacher/frontier | preview model | previewed, not fully available | n/a | seed: Meta | U | uncertain frontier stratum | TODO_VERIFY |
| 68 | Llama 3.3 70B Instruct | Meta | open-weight local | text | decoder-only | seed says 70B | public | seed: Meta/HF | M | open dense stratum | TODO_VERIFY |
| 69 | Llama 3.1 405B | Meta | open-weight local | text | decoder-only | seed says 405B | seed says 128K class | seed: Meta/HF | M* | open frontier dense basin | TODO_VERIFY |
| 70 | Llama 3.1 70B | Meta | open-weight local | text | decoder-only | seed says 70B | seed says 128K class | seed: Meta/HF | M | open dense stratum | TODO_VERIFY |
| 71 | Llama 3.1 8B | Meta | open-weight local | text | decoder-only | seed says 8B | seed says 128K class | seed: Meta/HF | M | open dense stratum | TODO_VERIFY |
| 72 | Llama Guard family | Meta | open-weight local | safety classifier | classifier/LLM | transformer classifier/LLM | public | seed: Meta/HF | M | safety/classifier basin | TODO_VERIFY |
| 73 | Mistral Medium 3.5 | Mistral AI | closed API | text/coding | closed/API model | not public | public docs | seed: Mistral docs | E | closed/API stratum | TODO_VERIFY |
| 74 | Mistral Small 4 | Mistral AI | open-weight API | multimodal instruct/reasoning/coding | multimodal transformer | public; hybrid single model | seed says 256K | seed: Mistral docs | M | multimodal open/API stratum | TODO_VERIFY |
| 75 | Magistral Medium | Mistral AI | closed API preview | reasoning | API reasoning model | not public | public API | seed: Mistral docs/paper | E | reasoning/API stratum | TODO_VERIFY |
| 76 | Magistral Small | Mistral AI | open-weight local | reasoning | decoder-only | public open reasoning model | public | seed: Mistral docs/paper | M | open reasoning stratum | TODO_VERIFY |
| 77 | Mistral Large | Mistral AI | closed API | text/reasoning | closed/API model | not public | public docs | seed: Mistral docs | E | closed/API stratum | TODO_VERIFY |
| 78 | Pixtral Large | Mistral AI | API/weights depending variant | vision-language | multimodal transformer | public for some variants | public | seed: Mistral docs | P | multimodal stratum | TODO_VERIFY |
| 79 | Codestral | Mistral AI | API/weights variant | code | decoder-only/API code model | public for some releases | public | seed: Mistral docs | P/M | coding stratum | TODO_VERIFY |
| 80 | Devstral Small | Mistral AI | open-weight local | coding agent | decoder/tool-oriented model | public open-weight | public | seed: Mistral docs | M | coding/open stratum | TODO_VERIFY |
| 81 | Mixtral 8x22B | Mistral AI | open-weight local | text/code | MoE | seed says 8x22B public | public | seed: Mistral/HF | M* | open MoE basin | TODO_VERIFY |
| 82 | Mixtral 8x7B | Mistral AI | open-weight local | text/code | MoE | seed says 8x7B public | public | seed: Mistral/HF | M | open MoE basin | TODO_VERIFY |
| 83 | Mistral 7B | Mistral AI | open-weight local | text | decoder-only | seed says 7B | public | seed: HF | M | research/open stratum | TODO_VERIFY |
| 84 | Voxtral Mini / Voxtral family | Mistral AI | open-weight API | speech/audio | audio/multimodal | public docs | n/a | seed: Mistral docs | P/M | audio stratum | TODO_VERIFY |
| 85 | Gemma 4 largest open variants | Google DeepMind | open-weight local | text+image, some audio | multimodal open | open-weight Gemma family | up to 256K / 128K small | seed: Google/HF | M | multimodal open basin | TODO_VERIFY |
| 86 | Gemma 4 E4B | Google DeepMind | open-weight local | efficient multimodal | multimodal open | open weights | public | seed: Google/HF | M | efficient/open stratum | TODO_VERIFY |
| 87 | Gemma 4 E2B | Google DeepMind | open-weight local | efficient multimodal | multimodal open | open weights | public | seed: Google/HF | M | efficient/open stratum | TODO_VERIFY |
| 88 | Gemma 3 27B | Google DeepMind | open-weight local | text+image | multimodal open | seed says 27B | seed says 128K | seed: Google/HF | M | multimodal open stratum | TODO_VERIFY |
| 89 | Gemma 3 12B | Google DeepMind | open-weight local | text+image | multimodal open | seed says 12B | seed says 128K | seed: Google/HF | M | multimodal open stratum | TODO_VERIFY |
| 90 | Gemma 3 4B | Google DeepMind | open-weight local | text+image | multimodal open | seed says 4B | seed says 128K | seed: Google/HF | M | multimodal open stratum | TODO_VERIFY |
| 91 | Gemma 3 1B | Google DeepMind | open-weight local | text | decoder-only | seed says 1B | seed says 128K | seed: Google/HF | M | small/open stratum | TODO_VERIFY |
| 92 | Phi-4 | Microsoft | open-weight local | text | decoder-only | open-weight Phi family | public card | seed: HF | M | open dense stratum | TODO_VERIFY |
| 93 | Phi-4-mini-instruct | Microsoft | open-weight local | text | decoder-only | lightweight open model | seed says 128K | seed: HF | M | small/open stratum | TODO_VERIFY |
| 94 | Phi-4-mini-reasoning | Microsoft | open-weight local | math/reasoning | decoder-only | lightweight open model | seed says 128K | seed: HF | M | reasoning/open stratum | TODO_VERIFY |
| 95 | Phi-4-reasoning-plus | Microsoft | open-weight local | reasoning | decoder-only | open-weight | seed says 64K tested extension | seed: HF | M | reasoning/open stratum | TODO_VERIFY |
| 96 | Phi-4-multimodal-instruct | Microsoft | open-weight local | text+image+audio input | multimodal open | open multimodal | seed says 128K | seed: HF | M | multimodal open stratum | TODO_VERIFY |
| 97 | Phi-4-reasoning-vision-15B | Microsoft | open-weight local | vision reasoning | multimodal open | seed says 15B and public components | public | seed: HF | M | multimodal open stratum | TODO_VERIFY |
| 98 | Pythia-12B | EleutherAI | open-weight local research | text | decoder-only | seed says 12B | public | seed: HF/paper | M++ | research-control basin | TODO_VERIFY |
| 99 | Pythia-6.9B | EleutherAI | open-weight local research | text | decoder-only | seed says 6.9B | public | seed: HF/paper | M++ | research-control basin | TODO_VERIFY |
| 100 | Pythia-2.8B / 1B / 410M / smaller checkpoints | EleutherAI | open-weight local research | text | decoder-only | controlled suite, many checkpoints | public | seed: HF/paper | M++ | research-control basin | TODO_VERIFY |

## 5. Recommended Publication Use

This table should support model-space feature schema design, experimental model selection, external/internal access distinctions, and architecture-family breadth.

It should not be used as a final leaderboard, a verified top-100 ranking, or a source of hidden metadata for closed systems.
