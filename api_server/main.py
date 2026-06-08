# api_server/main.py
from fastapi import FastAPI
from api_server.router import router

app = FastAPI(title="Sign Language AI Inference Server")

# 라우터 등록
app.include_router(router, prefix="/api")

# 실행 명령어: uvicorn main:app --reload
# uvicorn.run("api_server.main:app", host="127.0.0.1", port=8001, reload=False)