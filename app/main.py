from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from crud.base import create_db_and_tables
from routers import document_router
from routers.base import failure


app = FastAPI()
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
    create_db_and_tables()
    uvicorn.run("main:app", port=8082, log_level="info", reload=True)
