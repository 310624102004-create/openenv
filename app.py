"""app.py — Docker entrypoint for the CodeDebug OpenEnv server."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn
from openenv_env.server import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")
