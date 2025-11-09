from fastapi import FastAPI

app = FastAPI(title="Revenant v6", version="0.0.1")

@app.get("/")
def root():
    return {"status": "ok", "message": "Revenant Core Online"}

@app.get("/api/status")
def status():
    return {"uptime": "ok", "agents_loaded": 0}
