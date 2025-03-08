import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from crud.base import create_db_and_tables

from routers.base import failure

# 导入 fastApi 子模块
from routers import chat_router
from routers import chat_session_router
from routers import document_router

# FastAPI 主入口
app = FastAPI()

# 将 fastApi 子模块整合到 app 中
app.include_router(chat_router.router)
app.include_router(chat_session_router.router)
app.include_router(document_router.router)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """
    重写 fastApi 错误信息
    """
    return JSONResponse(
        failure(exc.status_code, exc.detail), status_code=exc.status_code
    )


if __name__ == "__main__":
    # 创建或启动数据库
    create_db_and_tables()
    # 启动 uvicorn 服务
    uvicorn.run("main:app", port=8082, log_level="info", reload=True)
