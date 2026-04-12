# Module 05 — Deployment

## Local Development

```bash
pip install -e ".[dev]"
uvicorn openenv_env.server:create_app --factory --port 8000 --reload
# or
python app.py
```

## Docker

```bash
# Build
docker build -t codedebug-env:latest .

# Run
docker run -p 7860:7860 codedebug-env:latest

# With env vars
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=hf_xxx \
  codedebug-env:latest
```

## Hugging Face Spaces

1. Create a new Space (type: Docker) at https://huggingface.co/spaces
2. Push your repo:
```bash
git remote add hf https://huggingface.co/spaces/<username>/codedebug-env
git push hf main
```
3. HF Spaces auto-builds your Dockerfile and exposes port 7860.
4. Confirm the space is live:
```bash
curl https://<username>-codedebug-env.hf.space/api/v1/health
```

## GitHub Actions CI

```yaml
name: Validate
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python validate_submission.py --minimal
```

## Pre-Submission Checklist
- [ ] python validate_submission.py --minimal passes all checks
- [ ] python inference.py emits [START], [STEP], [END] tags
- [ ] Docker build succeeds locally
- [ ] HF Space URL responds to /api/v1/health
- [ ] manifest.json has all required keys
- [ ] README is complete and accurate

## Troubleshooting

| Problem               | Fix |
|-----------------------|-----|
| Port already in use   | Change PORT env var |
| HF Space build fails  | Check requirements.txt for conflicts |
| Inference crashes     | Ensure API_BASE_URL and MODEL_NAME are set |
| Low rewards           | Review grader weights in graders.py |
