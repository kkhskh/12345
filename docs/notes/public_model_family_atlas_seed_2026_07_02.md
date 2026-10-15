3. Candidate top-100 model/system list

This is an atlas seed, not a final capability ranking. Order is grouped by family/source relevance.
Access categories: C/API = closed API or UI; OW/API = open/open-weight with hosted API; OW/local = weights available for local analysis; System = tool/RAG/agent wrapper; Media API = generative media system.
Mechanistic access: E = external-only; M = internal-mechanistic possible; P = partially possible; U = unknown.

#	Canonical model/system	Developer	Access	Modality	Public arch / params	Context	Release / license	Evidence source	Mechanistic access
1	GPT-5.6 preview	OpenAI	C/API limited	text+vision+tools	not public	not fully public	trusted preview	OpenAI docs	E
2	GPT-5.5	OpenAI	C/API	text+vision+tools	not public	1M	public API; priced	OpenAI, AA	E
3	GPT-5.5 Pro	OpenAI	C/API	reasoning/text+vision	not public	public priced tier	public API	OpenAI	E
4	GPT-5.4	OpenAI	C/API	text+vision+tools	not public	1M	public API	OpenAI	E
5	GPT-5.4 Pro	OpenAI	C/API	reasoning	not public	public priced tier	public API	OpenAI	E
6	GPT-5.4 mini	OpenAI	C/API	text+vision+tools	not public	400K	public API	OpenAI	E
7	GPT-5.4 nano	OpenAI	C/API	low-cost text/tool	not public	not fully reviewed	public API	OpenAI	E
8	GPT Image 2	OpenAI	Media API	image generation/editing	not public	n/a	public API	OpenAI pricing	E
9	GPT-Realtime-2	OpenAI	C/API	speech/audio realtime	not public	n/a	public API	OpenAI docs	E
10	Sora 2 Pro	OpenAI	Media API	video generation	not public	n/a	public API/priced	OpenAI pricing	E
11	Claude Fable 5	Anthropic	C/API	text+vision+tools	not public	1M	Jun 2026; priced	Anthropic, LMArena, AA	E
12	Claude Mythos 5	Anthropic	C/API limited	cyber/agent workflows	not public	same class as Fable docs	invitation-only	Anthropic docs	E
13	Claude Opus 4.8	Anthropic	C/API	text+vision+agentic coding	not public	1M	May 2026; priced	Anthropic, AA	E
14	Claude Opus 4.7	Anthropic	C/API	text+vision+reasoning	not public	public	public API	AA	E
15	Claude Sonnet 5	Anthropic	C/API	text+vision+coding	not public	1M	Jun 2026; priced	Anthropic docs	E
16	Claude Haiku 4.5	Anthropic	C/API	fast text+vision	not public	200K	public API	Anthropic docs	E
17	Claude Code / Claude agentic tooling	Anthropic	System	coding agent/tool system	system wrapper	depends on model	public product/API layer	Anthropic docs	E
18	Gemini 3.5 Flash	Google	C/API	text+image+video+audio/PDF input	not public	1,048,576 input	May 2026 update	Gemini docs, AA	E
19	Gemini 3.1 Pro Preview	Google	C/API preview	multimodal reasoning/coding	not public	1,048,576 input	Feb 2026 preview	Gemini docs	E
20	Gemini 3 Flash Preview	Google	C/API preview	multimodal	not public	public priced	preview	Gemini pricing	E
21	Gemini 3.1 Flash-Lite	Google	C/API	low-cost multimodal	not public	public docs	public API	Gemini docs	E
22	Gemini Deep Research Max	Google	System	research agent/RAG	system wrapper	n/a	public model family	Gemini docs	E
23	Gemini Computer Use Preview	Google	System	computer-use agent	not public	n/a	preview	Gemini docs	E
24	Gemini Antigravity Agent	Google	System	coding/agent system	not public	n/a	public docs	Gemini docs	E
25	Nano Banana Pro / Gemini image	Google	Media API	image generation	not public	n/a	public docs	Gemini docs	E
26	Veo 3.1 Preview	Google	Media API	video generation	not public	n/a	preview	Gemini docs	E
27	Gemini Robotics-ER 1.6 Preview	Google	C/API preview	robotics embodied reasoning	not public	n/a	preview	Gemini docs	E/U
28	Grok 4.3	xAI	C/API	text+tools	not public	1M	public API/priced	xAI docs	E
29	Grok Build 0.1	xAI	C/API	agentic coding	not public	256K	May 2026; priced	xAI docs	E
30	Grok 4.1 Fast / Grok long-context variants	xAI	C/API	text+tools	not public	public via docs/leaderboards	public API	xAI, AA	E
31	Perplexity Sonar	Perplexity	C/API/System	web-grounded QA	not public	not architecture-public	API/priced	Perplexity docs	E
32	Perplexity Sonar Pro	Perplexity	C/API/System	web-grounded QA	not public	not architecture-public	API/priced	Perplexity docs	E
33	Perplexity Sonar Reasoning Pro	Perplexity	C/API/System	reasoning+search	not public	not architecture-public	API/priced	Perplexity docs	E
34	Perplexity Sonar Deep Research	Perplexity	C/API/System	deep research+search	not public	not architecture-public	API/priced	Perplexity docs	E
35	Perplexity Agent API	Perplexity	System	tool-using web agent	wrapper over models/tools	bounded tool loop	public API	Perplexity docs	E/P
36	GLM-5.2 Max	Z.ai	OW/API or API	long-horizon coding/agent	open-source/open-weight per Z.ai/AA; exact row verify	1M	Jun 2026	Z.ai, AA	M/P
37	GLM-5.2	Z.ai	OW/API	long-context agentic	public docs	1M	Jun 2026	Z.ai docs	M/P
38	GLM-5.1	Z.ai	API/possibly OW	long-horizon agentic	public docs	not fully reviewed	Apr 2026	Z.ai docs	P/U
39	GLM-5	Z.ai	OW/local+API	text/coding agentic	public weights; MIT noted	public docs	Feb 2026; MIT	Z.ai docs	M
40	GLM-4.7	Z.ai	OW/API	coding/reasoning	public docs	not fully reviewed	public	Z.ai docs	M/P
41	DeepSeek V4 Pro	DeepSeek	OW/API	long-context reasoning/coding	MoE; 1.6T total / 49B active public	1M	Apr 2026	DeepSeek, HF, Reuters	M*
42	DeepSeek V4 Flash	DeepSeek	OW/API	efficient long-context	MoE; 284B total / 13B active public	1M class	Apr 2026	DeepSeek, NVIDIA/HF	M*
43	DeepSeek V3.2	DeepSeek	OW/API	reasoning/agents	public open model	public	Dec 2025	DeepSeek docs	M*
44	DeepSeek V3.2-Speciale	DeepSeek	OW/API	reasoning-first	public open model	public	Dec 2025	DeepSeek docs	M*
45	DeepSeek V3.1 Terminus	DeepSeek	OW/API	text/reasoning	public open model	public	2025	provider/HF	M*
46	DeepSeek V3	DeepSeek	OW/local	text/coding	MoE; 671B total / 37B active	public	public HF	HF	M*
47	DeepSeek R1	DeepSeek	OW/local+API	reasoning	public reasoning model	public	2025	HF	M*
48	DeepSeek R1-0528	DeepSeek	OW/local+API	reasoning update	public	public	2025	HF/provider	M*
49	DeepSeek-R1-Distill-Qwen-32B	DeepSeek/Qwen	OW/local	reasoning distill	dense Qwen-derived	public	open weights	HF	M
50	DeepSeek-R1-Distill-Llama-70B	DeepSeek/Meta	OW/local	reasoning distill	Llama-derived	public	open weights	HF	M
51	Qwen3.6-35B-A3B	Alibaba/Qwen	OW/local	text/reasoning	MoE; 35B total / 3B active indicated	public card	Apr 2026	HF	M
52	Qwen3.5-35B-A3B	Alibaba/Qwen	OW/local	text/reasoning	MoE; 35B total / 3B active indicated	public card	Mar 2026	HF	M
53	Qwen3.5-Plus	Alibaba/Qwen	C/API	agentic/text/tools	not fully public	1M default	Feb 2026	Qwen blog/Reuters	E/P
54	Qwen3-Max	Alibaba/Qwen	C/API	large agentic/coding	>1T public report	not fully reviewed	Sep 2025	Reuters/Qwen	E/P
55	Qwen3.5-Omni	Alibaba/Qwen	C/API/possibly OW variants	audio/vision/text	hybrid attention MoE; hundreds of B public in report	256K reported	2026 report	Qwen report	P/M if weights
56	Qwen3-235B-A22B	Alibaba/Qwen	OW/local	text/reasoning	MoE; 235B / 22B active	public	Apr 2025; Apache 2.0 family	Qwen	M*
57	Qwen3-30B-A3B	Alibaba/Qwen	OW/local	text/reasoning	MoE; 30B / 3B active	public	Apache 2.0 family	Qwen	M
58	Qwen3-32B	Alibaba/Qwen	OW/local	text/reasoning	dense 32B	public	Apache 2.0 family	HF/Qwen	M
59	Qwen3-14B	Alibaba/Qwen	OW/local	text/reasoning	dense 14B	public	Apache 2.0 family	HF/Qwen	M
60	Qwen3-8B	Alibaba/Qwen	OW/local	text/reasoning	dense 8B	public	May 2025; Apache 2.0 family	HF	M
61	Qwen2.5-VL-72B	Alibaba/Qwen	OW/local	vision-language	public VLM	public	open weights	HF/Qwen	M*
62	Qwen2.5-Coder-32B	Alibaba/Qwen	OW/local	code	decoder LLM	public	open weights	HF/Qwen	M
63	QwQ-32B	Alibaba/Qwen	OW/local	reasoning	decoder LLM	public	open weights	HF/Qwen	M
64	Qwen-Scope SAE artifacts	Alibaba/Qwen researchers	Research artifact	interpretability	SAEs for Qwen3/Qwen3.5 variants	n/a	public research	Qwen-Scope paper	M++
65	Llama 4 Scout	Meta	OW/local	native multimodal	MoE; 17B active / 16 experts public	very long, reported 10M	Apr 2025; Llama license	Meta/HF	M*
66	Llama 4 Maverick	Meta	OW/local	native multimodal	MoE; 17B active / 128 experts public	public	Apr 2025; Llama license	Meta/HF	M*
67	Llama 4 Behemoth preview	Meta	preview/not generally open	teacher/frontier	MoE; previewed, not fully available	n/a	preview	Meta	U
68	Llama 3.3 70B Instruct	Meta	OW/local	text	decoder-only, 70B	public	Llama license	Meta/HF	M
69	Llama 3.1 405B	Meta	OW/local	text	decoder-only, 405B	128K class	Llama license	Meta/HF	M*
70	Llama 3.1 70B	Meta	OW/local	text	decoder-only, 70B	128K class	Llama license	Meta/HF	M
71	Llama 3.1 8B	Meta	OW/local	text	decoder-only, 8B	128K class	Llama license	Meta/HF	M
72	Llama Guard family	Meta	OW/local	safety classifier	transformer classifier/LLM	public	Llama license	Meta/HF	M
73	Mistral Medium 3.5	Mistral AI	C/API	text/coding	not public	public docs	current model	Mistral docs	E
74	Mistral Small 4	Mistral AI	OW/API	multimodal instruct/reasoning/coding	public; hybrid single model	256K	Mar 2026; Apache 2.0	Mistral docs	M
75	Magistral Medium	Mistral AI	C/API preview	reasoning	not public	public API	Jun 2025	Mistral docs/paper	E
76	Magistral Small	Mistral AI	OW/local	reasoning	public open reasoning model	public	Jun 2025; Apache 2.0	Mistral docs/paper	M
77	Mistral Large	Mistral AI	C/API	text/reasoning	not public	public docs	public API	Mistral docs	E
78	Pixtral Large	Mistral AI	C/API or weights depending variant	vision-language	public for some variants	public	public model family	Mistral docs	P
79	Codestral	Mistral AI	API/weights variant	code	public for some releases	public	license varies	Mistral docs	P/M
80	Devstral Small	Mistral AI	OW/local	coding agent	public open-weight	public	open release	Mistral docs	M
81	Mixtral 8x22B	Mistral AI	OW/local	text/code	MoE; 8x22B public	public	open-weight	Mistral/HF	M*
82	Mixtral 8x7B	Mistral AI	OW/local	text/code	MoE; 8x7B public	public	Apache-style open release	Mistral/HF	M
83	Mistral 7B	Mistral AI	OW/local	text	decoder-only 7B	public	Apache 2.0	HF	M
84	Voxtral Mini / Voxtral family	Mistral AI	OW/API	speech/audio	public docs	n/a	public model family	Mistral docs	P/M
85	Gemma 4 largest open variants	Google DeepMind	OW/local	text+image, some audio	open-weight Gemma family	up to 256K / 128K small	recent model card	Google/HF	M
86	Gemma 4 E4B	Google DeepMind	OW/local	efficient multimodal	open weights	public	Gemma terms	Google/HF	M
87	Gemma 4 E2B	Google DeepMind	OW/local	efficient multimodal	open weights	public	Gemma terms	Google/HF	M
88	Gemma 3 27B	Google DeepMind	OW/local	text+image	open weights; 27B	128K	Gemma terms	Google/HF	M
89	Gemma 3 12B	Google DeepMind	OW/local	text+image	open weights; 12B	128K	Gemma terms	Google/HF	M
90	Gemma 3 4B	Google DeepMind	OW/local	text+image	open weights; 4B	128K	Gemma terms	Google/HF	M
91	Gemma 3 1B	Google DeepMind	OW/local	text	open weights; 1B	128K	Gemma terms	Google/HF	M
92	Phi-4	Microsoft	OW/local	text	open-weight Phi family	public card	Dec 2024	HF	M
93	Phi-4-mini-instruct	Microsoft	OW/local	text	lightweight open model	128K	Mar 2025	HF	M
94	Phi-4-mini-reasoning	Microsoft	OW/local	math/reasoning	lightweight open model	128K	Apr 2025	HF	M
95	Phi-4-reasoning-plus	Microsoft	OW/local	reasoning	open-weight	64K tested extension	Apr 2025	HF	M
96	Phi-4-multimodal-instruct	Microsoft	OW/local	text+image+audio input	open multimodal	128K	Mar 2025	HF	M
97	Phi-4-reasoning-vision-15B	Microsoft	OW/local	vision reasoning	15B; SigLIP-2 vision encoder + Phi backbone public	public	HF	HF	M
98	Pythia-12B	EleutherAI	OW/local research	text	decoder-only 12B	public	Apache 2.0 family	HF/paper	M++
99	Pythia-6.9B	EleutherAI	OW/local research	text	decoder-only 6.9B	public	Apache 2.0 family	HF/paper	M++
100	Pythia-2.8B / 1B / 410M / smaller checkpoints	EleutherAI	OW/local research	text	controlled suite, many checkpoints	public	Apache 2.0 family	HF/paper	M++

* M* = internal mechanistic work is possible but hardware-heavy.
M++ = unusually good for mechanistic research because weights/checkpoints/training controls or SAE artifacts are available.

Key source grounding for table groups: OpenAI docs list GPT-5.5/5.4 family, modalities, tools, context, and prices; Anthropic docs list Fable 5, Opus 4.8, Sonnet 5, Haiku 4.5, context, pricing, availability, and Mythos access limits; Google docs list Gemini 3.x models, multimodal inputs, tools, context windows, and pricing; xAI docs list Grok 4.3 and Grok Build public context/prices; Perplexity docs list Sonar models, pricing, API search/tool behavior; Meta describes Llama 4 Scout/Maverick as open-weight natively multimodal MoE models; Qwen and DeepSeek sources disclose open-weight releases and selected public parameter/context details; Mistral sources disclose Mistral Small 4, Magistral, and related open/API models; Gemma, Phi, and Pythia model cards/papers support open-weight and mechanistic feasibility claims.