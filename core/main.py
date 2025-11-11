from fastapi import FastAPI
from db.database import  engine, SessionLocal
from sqlalchemy import text
from db.database import Base


Base.metadata.create_all(bind=engine)
app = FastAPI(title="Revenant v6", version="0.0.1")

@app.get("/")
def root():
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT 1")).scalar()
        return {"status": "ok","message": "DB connected ✅", "result": result}
    finally:
        db.close()