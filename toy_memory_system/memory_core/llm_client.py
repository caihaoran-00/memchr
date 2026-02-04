"""
LLM客户端：支持多种提供商
"""
import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import httpx

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig


class LLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """发送对话请求"""
        pass
    
    @abstractmethod
    async def extract_json(
        self, 
        prompt: str, 
        schema_hint: str = ""
    ) -> Dict:
        """提取结构化JSON数据"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI兼容的客户端（支持OpenAI、Azure、其他兼容API）"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.api_key = config.llm_api_key
        self.base_url = config.llm_base_url or "https://api.openai.com/v1"
        self.model = config.llm_model
        self.extraction_model = config.extraction_model
        self.timeout = config.llm_timeout
        self.max_retries = config.llm_max_retries
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 500,
        model: str = None
    ) -> str:
        """发送对话请求"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM请求失败: {e}")
                await asyncio.sleep(1 * (attempt + 1))  # 指数退避
        
        return ""
    
    async def extract_json(
        self, 
        prompt: str, 
        schema_hint: str = ""
    ) -> Dict:
        """提取结构化JSON数据"""
        system_prompt = f"""你是一个信息提取助手。请从用户输入中提取关键信息，并以JSON格式返回。
只返回JSON，不要有其他文字。
{schema_hint}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.chat(
            messages, 
            temperature=0.1,  # 低温度保证稳定输出
            max_tokens=800,
            model=self.extraction_model
        )
        
        # 尝试解析JSON
        try:
            # 处理可能的markdown代码块
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {}


class MockLLMClient(LLMClient):
    """Mock客户端：用于测试，不调用真实API"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """模拟对话响应"""
        last_message = messages[-1]["content"] if messages else ""
        return f"这是对'{last_message[:20]}...'的模拟回复"
    
    async def extract_json(
        self, 
        prompt: str, 
        schema_hint: str = ""
    ) -> Dict:
        """模拟JSON提取"""
        # 简单的规则提取
        result = {
            "summary": prompt[:100] if len(prompt) > 100 else prompt,
            "keywords": [],
            "emotion": "neutral",
            "importance": 0.5,
            "facts": [],
            "profile_updates": {}
        }
        
        # 简单的关键词提取（基于常见词）
        keywords_candidates = ["喜欢", "讨厌", "想要", "名字", "岁", "学校", "朋友", "家"]
        for kw in keywords_candidates:
            if kw in prompt:
                result["keywords"].append(kw)
        
        return result


class ZhipuClient(LLMClient):
    """智谱AI客户端（国内替代方案）"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.api_key = config.llm_api_key
        self.base_url = config.llm_base_url or "https://open.bigmodel.cn/api/paas/v4"
        self.model = config.llm_model or "glm-4-flash"  # 使用便宜的flash模型
        self.timeout = config.llm_timeout
        self.max_retries = config.llm_max_retries
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """发送对话请求"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM请求失败: {e}")
                await asyncio.sleep(1 * (attempt + 1))
        
        return ""
    
    async def extract_json(
        self, 
        prompt: str, 
        schema_hint: str = ""
    ) -> Dict:
        """提取结构化JSON数据"""
        system_prompt = f"""你是一个信息提取助手。请从用户输入中提取关键信息，并以JSON格式返回。
只返回JSON，不要有其他文字。
{schema_hint}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.chat(messages, temperature=0.1, max_tokens=800)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {}


def create_llm_client(config: MemoryConfig) -> LLMClient:
    """工厂函数：创建LLM客户端"""
    provider = config.llm_provider.lower()
    
    if provider == "openai":
        return OpenAIClient(config)
    elif provider == "zhipu":
        return ZhipuClient(config)
    elif provider == "mock":
        return MockLLMClient(config)
    else:
        raise ValueError(f"不支持的LLM提供商: {provider}")
