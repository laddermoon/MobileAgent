"""
阿里云 GUI-Plus API 适配器
用于替代本地 VLLM 部署，直接调用阿里云 GUI-Plus API
"""
import abc
import time
import base64
import os
import math
from PIL import Image
from io import BytesIO
from openai import OpenAI
from typing import Any, Optional, Tuple
from qwen_vl_utils import smart_resize
from dotenv import load_dotenv

load_dotenv()

ERROR_CALLING_LLM = 'Error calling LLM'


def smart_size(imageheight, imagewidth, point, factor, min_pixels, max_pixels, vl_high_resolution_images=False):
    """
    将模型返回的坐标转换为原始图片坐标

    Args:
        imageheight: 原始图片高度
        imagewidth: 原始图片宽度
        point: 模型返回的坐标点，格式如 {"x": xxx, "y": xxx}
        factor: 缩放因子，通常为 28
        min_pixels: 最小像素数
        max_pixels: 最大像素数
        vl_high_resolution_images: 是否使用高分辨率模式

    Returns:
        (x, y): 目标对象相对于原图的坐标
    """
    height = imageheight
    width = imagewidth

    if vl_high_resolution_images:
        max_pixels = 16384 * 28 * 28

    # 将高度调整为factor的整数倍
    h_bar = round(height / factor) * factor
    # 将宽度调整为factor的整数倍
    w_bar = round(width / factor) * factor

    # 对图像进行缩放处理
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    abs_x1 = int(point["x"] / w_bar * width)
    abs_y1 = int(point["y"] / h_bar * height)
    return abs_x1, abs_y1


def pil_to_base64(image):
    """将 PIL Image 转换为 base64 编码"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_base64(image_path):
    """
    读取图片并转换为 base64 编码
    使用 smart_resize 进行图片预处理

    Returns:
        (base64_url, original_width, original_height)
    """
    dummy_image = Image.open(image_path)
    original_width, original_height = dummy_image.width, dummy_image.height
    MIN_PIXELS = 3136
    MAX_PIXELS = 10035200
    resized_height, resized_width = smart_resize(
        dummy_image.height,
        dummy_image.width,
        factor=28,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    dummy_image = dummy_image.resize((resized_width, resized_height))
    base64_url = f"data:image/png;base64,{pil_to_base64(dummy_image)}"
    return base64_url, original_width, original_height


class AliyunGUIPlusWrapper:
    """
    阿里云 GUI-Plus API 包装器
    兼容 Mobile-Agent-v3 的 GUIOwlWrapper 接口
    """

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "gui-plus",
        max_retry: int = 10,
        temperature: float = 0.0,
    ):
        """
        初始化阿里云 GUI-Plus 客户端

        Args:
            api_key: 阿里云 API Key，如果为 None 则从环境变量 APIKEY 读取
            base_url: API 端点地址
            model_name: 模型名称，默认 "gui-plus"
            max_retry: 最大重试次数
            temperature: 温度参数
        """
        if max_retry <= 0:
            max_retry = 10
            print('Max_retry must be positive. Reset it to 10')
        self.max_retry = min(max_retry, 10)
        self.temperature = temperature
        self.model = model_name
        
        # 如果没有提供 api_key，从环境变量读取
        if api_key is None:
            api_key = os.getenv("APIKEY")
            if api_key is None:
                raise ValueError("API Key not provided and APIKEY environment variable not set")
        
        self.bot = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60  # 阿里云 API 可能需要更长的超时时间
        )

    def convert_messages_format_to_openaiurl(self, messages):
        """
        将消息格式转换为 OpenAI URL 格式

        Args:
            messages: 原始消息列表

        Returns:
            (转换后的消息列表, 图片尺寸列表)
            图片尺寸列表格式: [(width, height, image_path), ...]
        """
        converted_messages = []
        image_sizes = []  # 存储每张图片的原始尺寸

        for message in messages:
            new_content = []
            for item in message['content']:
                if list(item.keys())[0] == 'text':
                    new_content.append({'type': 'text', 'text': item['text']})
                elif list(item.keys())[0] == 'image':
                    base64_url, orig_w, orig_h = image_to_base64(item['image'])
                    new_content.append({
                        'type': 'image_url',
                        'image_url': {'url': base64_url}
                    })
                    image_sizes.append((orig_w, orig_h, item['image']))
            converted_messages.append({
                'role': message['role'],
                'content': new_content
            })
        return converted_messages, image_sizes

    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """
        纯文本预测（不带图片）
        
        Args:
            text_prompt: 文本提示
            
        Returns:
            (响应文本, payload, 原始响应)
        """
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self,
        text_prompt: str,
        images: list,
        messages=None
    ) -> tuple[str, Optional[bool], Any, list]:
        """
        多模态预测（带图片）

        Args:
            text_prompt: 文本提示
            images: 图片路径列表
            messages: 可选的完整消息列表

        Returns:
            (响应文本, payload, 原始响应, 图片尺寸列表)
        """
        if messages is None:
            # 构建消息 payload
            payload = [
                {
                    "role": "user",
                    "content": [
                        {"text": text_prompt},
                    ]
                }
            ]

            # 添加图片
            for image in images:
                payload[0]['content'].append({
                    'image': image
                })
        else:
            payload = messages

        # 转换为 OpenAI URL 格式，并获取图片尺寸
        payload, image_sizes = self.convert_messages_format_to_openaiurl(payload)

        # 重试机制
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS

        while counter > 0:
            try:
                # 调用阿里云 GUI-Plus API
                chat_completion = self.bot.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=self.temperature
                )

                response_text = chat_completion.choices[0].message.content
                return (response_text, payload, chat_completion, image_sizes)

            except Exception as e:
                counter -= 1
                if counter > 0:
                    print(f'Error calling Aliyun GUI-Plus API, retrying in {wait_seconds}s...')
                    print(f'Error: {e}')
                    time.sleep(wait_seconds)
                    wait_seconds *= 1.5  # 指数退避
                else:
                    print(f'Failed after {self.max_retry} retries')
                    print(f'Last error: {e}')

        return ERROR_CALLING_LLM, None, None, []


# 为了兼容性，创建别名
GUIOwlWrapper = AliyunGUIPlusWrapper
