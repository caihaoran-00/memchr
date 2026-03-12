"""
一键体验记忆系统。
"""

import asyncio
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig
from memory_core.manager import MemoryManager


async def main() -> None:
    demo_data_dir = "./demo_data"
    shutil.rmtree(demo_data_dir, ignore_errors=True)
    config = MemoryConfig(
        data_dir=demo_data_dir,
        conversation_turns_before_extraction=3,
    )

    manager = MemoryManager(config)
    user_id = "demo_child"

    print("=== Demo Start ===")
    print(f"provider: {config.llm_provider}")
    print(f"embedding_provider: {config.embedding_provider}")
    print(f"rerank_provider: {config.rerank_provider}")
    print(f"turns_before_extraction: {config.conversation_turns_before_extraction}")

    session = manager.start_session(user_id)
    turns = [
        ("user", "我叫小明，我5岁了。"),
        ("assistant", "你好呀，小明。你最近喜欢玩什么？"),
        ("user", "我最喜欢霸王龙和恐龙故事。"),
        ("assistant", "太棒了，霸王龙真的很酷。"),
        ("user", "我今天还画了一只霸王龙。"),
        ("assistant", "听起来很棒，我会记住你喜欢霸王龙和恐龙故事。"),
    ]

    for role, content in turns:
        manager.add_message(session.session_id, role, content)
        print(f"{role}: {content}")

    episode = await manager.end_session(session.session_id, extract_memory=True)
    print("\n=== Extracted Episode ===")
    if episode:
        print(episode.summary)
    else:
        print("no episode")

    profile = manager.get_user_profile(user_id)
    print("\n=== Profile ===")
    print(profile.to_dict() if profile else {})

    facts = manager.storage.get_facts(user_id)
    print("\n=== Facts ===")
    for fact in facts:
        print("-", fact.to_natural_language())

    recall_session = manager.start_session(user_id)
    manager.add_message(recall_session.session_id, "user", "你还记得我最喜欢什么吗？")
    context = await manager.get_memory_context(
        recall_session.session_id,
        "我最喜欢什么恐龙",
    )

    print("\n=== Recall Context ===")
    print(context.to_system_prompt())

    await manager.end_session(recall_session.session_id, extract_memory=False)
    print("\n=== Demo Done ===")


if __name__ == "__main__":
    asyncio.run(main())
