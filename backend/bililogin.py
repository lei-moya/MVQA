import requests
import json
import time
import base64
from io import BytesIO
import qrcode


class BiliClient:
    def __init__(self, sessdata=""):
        self.sessdata = sessdata
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.bilibili.com"
        }
        if sessdata:
            self.headers["Cookie"] = f"SESSDATA={sessdata}"

    def simple_get(self, url, params=None):
        if params is None:
            params = {}
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()  # 检查HTTP状态码
            return response
        except requests.exceptions.RequestException as e:
            # 捕获网络异常并返回一个带有错误信息的假响应对象
            class FakeResponse:
                def json(self):
                    return {"code": -1, "message": f"网络错误: {str(e)}"}
            return FakeResponse()

    def get_qr_info(self):
        """获取登录二维码信息"""
        url = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
        response = self.simple_get(url)
        data = response.json()
        if data["code"] != 0:
            # 抛出异常，因为get_qr_code路由会捕获并返回错误信息
            raise Exception(data.get("message", "获取二维码失败"))
        return data["data"]

    def get_qr_status(self, qr_key):
        """获取二维码状态"""
        url = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
        params = {"qrcode_key": qr_key}
        # 直接使用requests.get，而不是simple_get，因为需要获取cookies
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()  # 检查HTTP状态码
            data = response.json()
        except requests.exceptions.RequestException as e:
            # 网络错误
            return {"code": -1, "message": f"网络错误: {str(e)}"}, None
        
        if data["code"] != 0:
            # 返回错误状态而不是抛出异常
            return {"code": data["code"], "message": data.get("message", "获取二维码状态失败")}, None

        qr_status = data["data"]
        sessdata = None
        if qr_status["code"] == 0:
            # 登录成功，从cookie中获取SESSDATA
            try:
                # 尝试从response.cookies中获取
                sessdata = response.cookies.get("SESSDATA")
            except Exception:
                pass  # 忽略cookie解析错误
            
            # 如果从cookie中获取不到SESSDATA，尝试从响应数据中获取
            if not sessdata and "cookie_info" in qr_status:
                cookie_info = qr_status["cookie_info"]
                if "cookies" in cookie_info:
                    for cookie in cookie_info["cookies"]:
                        if cookie.get("name") == "SESSDATA":
                            sessdata = cookie.get("value")
                            break

        return qr_status, sessdata

    def check_login(self):
        """检查是否已登录"""
        if not self.sessdata:
            return False

        url = "https://api.bilibili.com/x/space/myinfo"
        response = self.simple_get(url)
        data = response.json()
        return data["code"] == 0


def generate_qr_code(url):
    """生成二维码图片"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def login():
    """执行登录流程"""
    client = BiliClient()

    # 1. 获取二维码信息
    qr_info = client.get_qr_info()
    qr_url = qr_info["url"]
    qr_key = qr_info["qrcode_key"]

    # 2. 生成并显示二维码
    img_base64 = generate_qr_code(qr_url)

    # 3. 轮询二维码状态
    while True:
        time.sleep(2)  # 每2秒检查一次
        qr_status, sessdata = client.get_qr_status(qr_key)

        if qr_status["code"] == 86038:
            # 二维码已过期
            return None
        elif qr_status["code"] == 0:
            # 登录成功
            return sessdata


def main():
    """主函数"""
    sessdata = login()
    if sessdata:
        # 验证登录状态
        client = BiliClient(sessdata)
        client.check_login()


if __name__ == "__main__":
    main()
