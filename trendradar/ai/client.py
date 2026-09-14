# coding=utf-8
"""
AI 客户端模块

基于 LiteLLM 的统一 AI 模型接口
支持 100+ AI 提供商（OpenAI、DeepSeek、Gemini、Claude、国内模型等）
"""

import os
from typing import Any, Dict, List

from litellm import completion


class AIClient:
    """统一的 AI 客户端（基于 LiteLLM）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 AI 客户端

        Args:
            config: AI 配置字典
                - MODEL: 模型标识（格式: provider/model_name）
                - API_KEY: API 密钥
                - API_BASE: API 基础 URL（可选）
                - TEMPERATURE: 采样温度
                - MAX_TOKENS: 最大生成 token 数
                - TIMEOUT: 请求超时时间（秒）
                - NUM_RETRIES: 重试次数（可选）
                - FALLBACK_MODELS: 备用模型列表（可选）
        """
        self.model = config.get("MODEL", "deepseek/deepseek-chat")
        self.api_key = config.get("API_KEY") or os.environ.get("AI_API_KEY", "")
        self.api_base = config.get("API_BASE", "")
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.max_tokens = config.get("MAX_TOKENS", 5000)
        self.timeout = config.get("TIMEOUT", 120)
        self.num_retries = config.get("NUM_RETRIES", 2)
        self.fallback_models = config.get("FALLBACK_MODELS", [])

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        调用 AI 模型进行对话。
    
        DeepSeek V4 兼容：
        1. 默认关闭 thinking，避免 reasoning 消耗 max_tokens 后 content 为空
        2. HTTP 200 但 content 为空时主动重试
        3. 支持 response_format 等额外参数
        """
    
        import time
    
        # 空响应单独重试。
        # LiteLLM num_retries 主要处理异常，不处理 HTTP 200 + content=""。
        empty_response_retries = kwargs.pop("empty_response_retries", 2)
    
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": kwargs.get("timeout", self.timeout),
            "num_retries": kwargs.get("num_retries", self.num_retries),
        }
    
        if self.api_key:
            params["api_key"] = self.api_key
    
        if self.api_base:
            params["api_base"] = self.api_base
    
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens and max_tokens > 0:
            params["max_tokens"] = max_tokens
    
        if self.fallback_models:
            params["fallbacks"] = self.fallback_models
    
        # ==========================================================
        # DeepSeek V4
        #
        # V4 默认开启 Thinking。
        # TrendRadar 的分类、翻译、摘要属于结构化任务，
        # 不需要长 reasoning。
        #
        # 显式关闭可以：
        # - 减少 Token
        # - 避免 reasoning_content 占满输出预算
        # - 提高 JSON 输出稳定性
        # ==========================================================
        if self.model.startswith("deepseek/"):
        
            params["extra_body"] = {
                **params.get("extra_body", {}),
                "thinking": {
                    "type": "disabled"
                }
            }
    
        # 其余 kwargs
        for key, value in kwargs.items():
        
            if key in {
                "temperature",
                "timeout",
                "num_retries",
                "max_tokens",
                "empty_response_retries",
            }:
                continue
        
            if key == "extra_body":
                params["extra_body"] = {
                    **params.get("extra_body", {}),
                    **value,
                }
            else:
                params[key] = value
    
        last_finish_reason = None
    
        for attempt in range(empty_response_retries + 1):
    
            response = completion(**params)
    
            choice = response.choices[0]
            message = choice.message
    
            content = message.content
    
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", str(item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in content
                )
    
            if content and str(content).strip():
                return str(content).strip()
    
            # ------------------------------------------------------
            # HTTP 请求成功，但模型没有生成最终 content
            # ------------------------------------------------------
    
            last_finish_reason = getattr(choice, "finish_reason", None)
    
            reasoning_content = getattr(
                message,
                "reasoning_content",
                None
            )
    
            reasoning_length = (
                len(reasoning_content)
                if isinstance(reasoning_content, str)
                else 0
            )
    
            print(
                f"[AI] 模型返回空 content "
                f"(第 {attempt + 1}/{empty_response_retries + 1} 次, "
                f"finish_reason={last_finish_reason}, "
                f"reasoning_length={reasoning_length})"
            )
    
            if attempt < empty_response_retries:
                # 1 秒、2 秒
                time.sleep(2 ** attempt)
    
        raise RuntimeError(
            "AI 连续返回空响应"
            f"（finish_reason={last_finish_reason}）"
        )

    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效

        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not self.model:
            return False, "未配置 AI 模型（model）"

        if not self.api_key:
            return False, "未配置 AI API Key，请在 config.yaml 或环境变量 AI_API_KEY 中设置"

        # 验证模型格式（应该包含 provider/model）
        if "/" not in self.model:
            return False, f"模型格式错误: {self.model}，应为 'provider/model' 格式（如 'deepseek/deepseek-chat'）"

        return True, ""
