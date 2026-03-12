"""
真实 Qwen 冒烟测试脚本。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigPresets
from memory_core.manager import MemoryManager


async def main() -> None:
    config = ConfigPresets.full_featured()
    manager = MemoryManager(config)

    session = manager.start_session("real_qwen_user_live")
    turns = [
        ("user", "我叫小明，我5岁了，我最喜欢恐龙。"),
        ("assistant", "太好了，你最喜欢哪种恐龙？"),
        ("user", "我最喜欢霸王龙。"),
        ("assistant", "霸王龙真的很酷。"),
        ("user", "我还喜欢听恐龙故事。"),
        ("assistant", "我记住了，你喜欢霸王龙和恐龙故事。"),
        ("user", "我今天还画了一只恐龙。"),
        ("assistant", "听起来很棒，你画的是霸王龙吗？"),
        ("user", "对，是霸王龙。"),
        ("assistant", "好的，我会记住你喜欢霸王龙。"),
    ]
    for role, content in turns:
        manager.add_message(session.session_id, role, content)

    episode = await manager.end_session(session.session_id, extract_memory=True)
    print("EPISODE_OK", bool(episode and episode.summary))
    print("EPISODE_SUMMARY", episode.summary if episode else "")

    session2 = manager.start_session("real_qwen_user_live")
    manager.add_message(session2.session_id, "user", "你还记得我最喜欢什么恐龙吗？")
    context = await manager.get_memory_context(session2.session_id, "我最喜欢什么恐龙")

    print("PROFILE_NAME", context.user_profile.name if context.user_profile else "")
    print("EPISODE_COUNT", len(context.relevant_episodes))
    print("FACT_COUNT", len(context.relevant_facts))
    if context.relevant_episodes:
        print("TOP_EPISODE", context.relevant_episodes[0].summary)
    if context.relevant_facts:
        print("TOP_FACT", context.relevant_facts[0].to_natural_language())


if __name__ == "__main__":
    asyncio.run(main())
