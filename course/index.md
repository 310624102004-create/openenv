# CodeDebug OpenEnv — Course Index

A five-module curriculum that takes you from OpenEnv fundamentals to deploying
a production-grade reinforcement-learning environment.

| Module | Topic | What You'll Learn |
|--------|-------|-------------------|
| [01 — Intro to OpenEnv](module_01_intro/README.md) | Framework overview | reset/step/state lifecycle, manifest, scoring |
| [02 — Environment Design](module_02_environment/README.md) | Building the env | Typed models, graders, episode flow |
| [03 — LLM Inference](module_03_inference/README.md) | Calling LLM APIs | OpenAI client, structured output, env vars |
| [04 — Grader Engineering](module_04_graders/README.md) | Reward shaping | Writing tight graders, composite scoring |
| [05 — Deployment](module_05_deployment/README.md) | Ship it | Docker, Hugging Face Spaces, CI |

---

## Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`
- Basic familiarity with FastAPI and Pydantic

## How to Use
Each module has a `README.md` (concepts) and a `notebook.ipynb` (hands-on exercises).
Work through them in order, or jump to whichever topic you need.
