"""
集成示例。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig
from memory_core.llm_client import create_llm_client
from memory_core.manager import MemoryManager


class SmartToyWithMemory:
    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self.memory = MemoryManager(self.config)
        self.llm = create_llm_client(self.config)
        self.base_prompt = (
            "你是一个友好的儿童玩具助手。"
            "请结合用户画像、相关事实和历史片段，用温和自然的方式回复。"
        )

    async def chat(self, user_id: str, session_id: str, user_input: str) -> str:
        self.memory.add_message(session_id, "user", user_input)
        context = await self.memory.get_memory_context(session_id, user_input)

        system_prompt = self.base_prompt
        memory_prompt = context.to_system_prompt()
        if memory_prompt:
            system_prompt += f"\n\n这是你可以参考的记忆信息：\n{memory_prompt}"

        messages = [{"role": "system", "content": system_prompt}]
        if context.working_memory:
            for message in context.working_memory.get_recent(5):
                messages.append({"role": message.role.value, "content": message.content})

        response = await self.llm.chat(messages, temperature=0.7, max_tokens=300)
        self.memory.add_message(session_id, "assistant", response)
        return response

    def start_conversation(self, user_id: str) -> str:
        return self.memory.start_session(user_id).session_id

    async def end_conversation(self, session_id: str) -> dict:
        episode = await self.memory.end_session(session_id, extract_memory=True)
        return {
            "success": True,
            "summary": episode.summary if episode else None,
        }


async def main() -> None:
    toy = SmartToyWithMemory()
    session_id = toy.start_conversation("integration_demo_user")

    reply = await toy.chat("integration_demo_user", session_id, "我叫小明，我喜欢恐龙。")
    print("assistant:", reply)

    result = await toy.end_conversation(session_id)
    print("conversation_summary:", result["summary"])


if __name__ == "__main__":
    asyncio.run(main())
