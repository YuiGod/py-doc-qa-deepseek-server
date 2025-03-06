def success(data: dict | None = None, message: str = "响应成功！"):
    """
    响应成功结果值
    """
    result = {"code": 200, "message": message, "data": data}
    return result


def failure(code: int, message: str = ""):
    """
    响应失败结果值
    """
    return {"code": code, "message": message}
