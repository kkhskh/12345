# Top 100 Model Maps V0

Status: seed maps from `docs/notes/public_model_family_atlas_seed_2026_07_02.md`. These are not final leaderboard claims, and all model metadata remains `TODO_VERIFY` unless separately source-verified.

## 1. Provider / Family Map

```mermaid
flowchart LR
    A[Top-100 seed atlas] --> OpenAI[OpenAI\nrows 1-10\n10 systems]
    A --> Anthropic[Anthropic\nrows 11-17\n7 systems]
    A --> Google[Google Gemini\nrows 18-27\n10 systems]
    A --> xAI[xAI\nrows 28-30\n3 systems]
    A --> Perplexity[Perplexity\nrows 31-35\n5 systems]
    A --> Zai[Z.ai GLM\nrows 36-40\n5 systems]
    A --> DeepSeek[DeepSeek\nrows 41-50\n10 systems]
    A --> Qwen[Alibaba Qwen\nrows 51-64\n14 systems/artifacts]
    A --> Meta[Meta Llama\nrows 65-72\n8 systems]
    A --> Mistral[Mistral AI\nrows 73-84\n12 systems]
    A --> Gemma[Google DeepMind Gemma\nrows 85-91\n7 systems]
    A --> Phi[Microsoft Phi\nrows 92-97\n6 systems]
    A --> Pythia[EleutherAI Pythia\nrows 98-100\n3 systems]
```

## 2. External vs Internal Map

```mermaid
flowchart TB
    subgraph External["External-observable map"]
        Closed[Closed APIs / UIs]
        Media[Media APIs]
        Systems[RAG / tool / agent systems]
        Benchmarks[Benchmarks, outputs, pricing, latency, context]
    end

    subgraph Internal["Internal obstruction map"]
        Open[Open/open-weight local models]
        Research[Research-control suites]
        SAE[SAE feature artifacts]
        Mechanistic[Logits, activations, patching, transports, obstruction metrics]
    end

    Closed --> Benchmarks
    Media --> Benchmarks
    Systems --> Benchmarks
    Open --> Mechanistic
    Research --> Mechanistic
    SAE --> Mechanistic
    Systems -. decompose scaffold/tool/retrieval decision first .-> Mechanistic
```

## 3. Mechanistic Feasibility Strata

```mermaid
flowchart LR
    E[E: external-only\nclosed frontier/media/system rows] --> P[P: partial access\nAPI/open-weight mixed or system rows]
    P --> M[M: internal possible\nopen dense/small rows]
    M --> MS[M*: hardware-heavy\nfrontier-scale open/MoE rows]
    M --> MPP[M++: research-control / SAE artifact rows]
```

## 4. Architecture / System Strata

```mermaid
mindmap
  root((Top-100 seed strata))
    Closed frontier API
      OpenAI
      Anthropic
      Google Gemini
      xAI
    RAG / agent systems
      Perplexity
      Gemini agents
      Claude tooling
    Open frontier MoE
      DeepSeek
      Qwen
      Llama
      Mixtral
    Open dense generalist
      Llama
      Qwen
      Mistral
      Gemma
      Phi
    Multimodal open
      Llama
      Qwen VL
      Gemma
      Phi
      Mistral
    Generative media
      OpenAI media
      Google media
    Research-control / SAE
      Pythia
      Qwen-Scope SAE
```
