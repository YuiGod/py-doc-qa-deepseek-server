import os
from pathlib import Path
import uuid
import aiofiles
import aiofiles.os
from fastapi import File, HTTPException, UploadFile
from sqlmodel import Session, func, select
from models.document_model import (
    Document,
    DocumentParams,
    DocumentRecord,
    DocumentUpdate,
    UpdateFormData,
    UploadFormData,
)
from .base import engine


class DocumentCrud:
    __BASE_PATH = "/home/ly/Project"
    __FILE_PATH = "/fileStorage"

    async def __save_file(self, file: UploadFile = File()):
        """I/O操作，写入文件"""

        oldFileName = os.path.basename(file.filename)
        # 生成唯一文件名称
        newFileName = f"{uuid.uuid4()}{Path(oldFileName).suffix}"
        savePath = Path(self.__BASE_PATH + self.__FILE_PATH) / newFileName
        newPath = Path(self.__FILE_PATH) / newFileName
        newFilePath = str(newPath)
        newSuffix = newPath.suffix

        # 利用 aiofiles 保存文件到指定目录
        async with aiofiles.open(savePath, "wb") as buffer:
            while content := await file.read(1024000):  # 每次读取 1KB
                await buffer.write(content)
        return newFileName, newFilePath, newSuffix

    async def __delete_file(self, path: str):
        """I/O操作，删除文件"""
        if await aiofiles.os.path.exists(path):
            await aiofiles.os.remove(path)

    async def add(self, data: UploadFormData):
        """新增文档记录"""
        file = data.file
        newFileName, newFilePath, newSuffix = await self.__save_file(file)
        doc = DocumentUpdate(
            name=data.name,
            file_name=newFileName,
            file_path=newFilePath,
            suffix=newSuffix,
            vector="",
        )

        with Session(engine) as session:
            db_document = Document.model_validate(doc)
            session.add(db_document)
            session.commit()
            session.refresh(db_document)
            return db_document

    async def update(self, data: UpdateFormData):
        """修改文档记录"""
        file = data.file
        doc = DocumentUpdate(
            name=data.name,
            file_name=data.file_name,
            file_path=data.file_path,
            vector=data.vector,
        )

        with Session(engine) as session:
            db_document = session.get(Document, data.id)
            if not db_document:
                raise HTTPException(status_code=404, detail="找不到该记录。")

            if file:
                # 数据库查找旧的文件名，并删除文件
                path = Path(self.__BASE_PATH + db_document.file_path)
                await self.__delete_file(path)
                # 保存新的文件
                newFileName, newFilePath, newSuffix = await self.__save_file(file)
                doc.file_name = newFileName
                doc.file_path = newFilePath
                doc.suffix = newSuffix
                doc.vector = ""

            doc = doc.model_dump(exclude_unset=True)
            db_document.sqlmodel_update(doc)
            session.add(db_document)
            session.commit()
            session.refresh(db_document)
            return db_document

    async def delete(self, id: str):
        """删除文档记录，并删除文件"""
        with Session(engine) as session:
            document = session.get(Document, id)
            if not document:
                raise HTTPException(status_code=404, detail="文档未找到")

            path = Path(self.__BASE_PATH + document.file_path)
            await self.__delete_file(path)

            session.delete(document)
            session.commit()

    def page(self, params: DocumentParams):
        """查询文档记录，分页"""
        offset = params.page_num * params.page_size
        limit = params.page_size
        name = params.name
        with Session(engine) as session:
            count_query = select(func.count(Document.id))
            query = select(Document)

            # 根据 name 模糊查询，并查询总数
            if name:
                count_query = count_query.where(Document.name.contains(name))
                query = query.where(Document.name.contains(name))

            query.order_by(Document.date).offset(offset).limit(limit)

            total = session.exec(count_query).one()
            items = session.exec(query).all()
            list = [DocumentRecord.model_validate(d) for d in items]
            return {
                "total": total,
                "page_num": params.page_num,
                "page_size": params.page_size,
                "list": list,
            }

    def download(self, item_id: str):
        with Session(engine) as session:
            document = session.get(Document, item_id)
            if not document:
                raise HTTPException(status_code=404, detail="文档未找到")

            file_path = Path(self.__BASE_PATH + document.file_path)
            realName = (
                str(document.name + file_path.suffix).encode("utf-8").decode("latin1")
            )

            return file_path, realName

    def vector_all_docs(self):
        with Session(engine) as session:
            query = select(Document)
            docList = session.exec(query).all()
            for doc in docList:
                doc.vector = "yes"
                session.add(doc)
            session.commit()
