"""
集成示例：展示如何将记忆系统集成到智能玩具主程序
这是一个完整的对话循环示例
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig, ConfigPresets
from memory_core.manager import MemoryManager
from memory_core.llm_client import create_llm_client


class SmartToyWithMemory:
    """带记忆功能的智能玩具示例"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or ConfigPresets.balanced()
        self.memory = MemoryManager(self.config)
        self.llm = create_llm_client(self.config)
        
        # 基础系统提示词
        self.base_prompt = """你是一个可爱友好的智能玩具助手，专门陪伴小朋友聊天和学习。
你的性格特点：
- 温暖、耐心、充满好奇心
- 说话简单易懂，适合儿童
- 会主动关心小朋友的感受
- 喜欢讲故事和玩游戏
"""
    
    async def chat(self, user_id: str, session_id: str, user_input: str) -> str:
        """
        处理用户输入并返回回复
        
        这个方法展示了记忆系统的完整使用流程：
        1. 记录用户消息
        2. 检索相关记忆
        3. 构建增强的prompt
        4. 调用LLM生成回复
        5. 记录助手回复
        """
        
        # 1. 记录用户消息到工作记忆
        self.memory.add_message(session_id, "user", user_input)
        
        # 2. 获取记忆上下文
        context = self.memory.get_memory_context(session_id, user_input)
        
        # 3. 构建增强的系统提示词
        memory_prompt = context.to_system_prompt()
        full_system_prompt = self.base_prompt
        
        if memory_prompt:
            full_system_prompt += f"\n\n{memory_prompt}\n\n请根据以上信息，更个性化地回应用户。"
        
        # 4. 构建对话历史
        messages = [{"role": "system", "content": full_system_prompt}]
        
        # 添加工作记忆中的对话历史
        if context.working_memory:
            for msg in context.working_memory.get_recent(5):
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        # 5. 调用LLM
        try:
            response = await self.llm.chat(messages, temperature=0.8, max_tokens=300)
        except Exception as e:
            response = f"哎呀，我有点累了，让我休息一下再聊吧！（错误：{e}）"
        
        # 6. 记录助手回复
        self.memory.add_message(session_id, "assistant", response)
        
        return response
    
    def start_conversation(self, user_id: str) -> str:
        """开始新对话"""
        session = self.memory.start_session(user_id)
        return session.session_id
    
    async def end_conversation(self, session_id: str) -> dict:
        """结束对话并提取记忆"""
        episode = await self.memory.end_session(session_id, extract_memory=True)
        
        if episode:
            return {
                "success": True,
                "summary": episode.summary,
                "keywords": episode.keywords,
                "emotion": episode.emotion
            }
        return {"success": True, "summary": None}
    
    def get_user_profile(self, user_id: str) -> dict:
        """获取用户画像"""
        profile = self.memory.get_user_profile(user_id)
        if profile:
            return {
                "name": profile.name,
                "age": profile.age,
                "tags": profile.tags
            }
        return {}


async def interactive_demo():
    """交互式演示"""
    print("=" * 60)
    print("  智能玩具记忆系统 - 交互式演示")
    print("=" * 60)
    print("\n提示：")
    print("- 输入 'quit' 退出")
    print("- 输入 'new' 开始新会话")
    print("- 输入 'profile' 查看用户画像")
    print("- 输入 'stats' 查看统计信息")
    print("-" * 60)
    
    # 使用Mock配置进行演示（不调用真实API）
    config = ConfigPresets.minimal()
    config.data_dir = "./demo_data"
    
    toy = SmartToyWithMemory(config)
    user_id = "demo_child"
    session_id = toy.start_conversation(user_id)
    
    print(f"\n[系统] 会话已开始，用户ID: {user_id}")
    print(f"[系统] 会话ID: {session_id}\n")
    
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\n[系统] 正在结束会话并提取记忆...")
            result = await toy.end_conversation(session_id)
            if result.get("summary"):
                print(f"[系统] 记忆摘要: {result['summary']}")
            print("[系统] 再见！")
            break
        
        elif user_input.lower() == 'new':
            print("\n[系统] 结束当前会话...")
            await toy.end_conversation(session_id)
            session_id = toy.start_conversation(user_id)
            print(f"[系统] 新会话已开始: {session_id}\n")
            continue
        
        elif user_input.lower() == 'profile':
            profile = toy.get_user_profile(user_id)
            print(f"\n[用户画像] {profile}\n")
            continue
        
        elif user_input.lower() == 'stats':
            stats = toy.memory.get_stats(user_id)
            print(f"\n[统计信息] {stats}\n")
            continue
        
        # 正常对话
        response = await toy.chat(user_id, session_id, user_input)
        print(f"玩具: {response}\n")


async def batch_demo():
    """批量演示：模拟多次对话积累记忆"""
    print("=" * 60)
    print("  智能玩具记忆系统 - 批量对话演示")
    print("=" * 60)
    
    config = ConfigPresets.minimal()
    config.data_dir = "./demo_data"
    os.makedirs(config.data_dir, exist_ok=True)
    
    toy = SmartToyWithMemory(config)
    user_id = "batch_demo_child"
    
    # 模拟多次对话
    conversations = [
        # 第一次对话：自我介绍
        [
            "你好呀！",
            "我叫豆豆",
            "我今年6岁了",
            "我喜欢画画",
        ],
        # 第二次对话：聊兴趣
        [
            "我今天画了一幅画",
            "画的是恐龙！我最喜欢霸王龙",
            "它有大大的牙齿",
        ],
        # 第三次对话：聊学校
        [
            "我今天去上学了",
            "我有一个好朋友叫小花",
            "我们一起玩跳绳",
        ],
    ]
    
    for i, conv in enumerate(conversations):
        print(f"\n--- 对话 {i+1} ---")
        session_id = toy.start_conversation(user_id)
        
        for msg in conv:
            print(f"用户: {msg}")
            response = await toy.chat(user_id, session_id, msg)
            print(f"玩具: {response}")
        
        result = await toy.end_conversation(session_id)
        if result.get("summary"):
            print(f"\n[记忆摘要] {result['summary']}")
    
    # 查看累积的用户画像
    print("\n" + "=" * 60)
    print("最终用户画像：")
    profile = toy.get_user_profile(user_id)
    print(f"  名字: {profile.get('name', '未知')}")
    print(f"  年龄: {profile.get('age', '未知')}")
    print(f"  标签: {profile.get('tags', [])}")
    
    print("\n统计信息：")
    stats = toy.memory.get_stats(user_id)
    print(f"  情景记忆数: {stats['episode_count']}")
    print(f"  知识事实数: {stats['fact_count']}")
    
    # 测试新会话中的记忆召回
    print("\n" + "=" * 60)
    print("测试记忆召回：开始新对话")
    session_id = toy.start_conversation(user_id)
    toy.memory.add_message(session_id, "user", "你还记得我叫什么吗？")
    
    context = toy.memory.get_memory_context(session_id, "记得我")
    print("\n记忆上下文：")
    print("-" * 40)
    print(context.to_system_prompt())
    print("-" * 40)
    
    await toy.end_conversation(session_id)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        asyncio.run(batch_demo())
    else:
        asyncio.run(interactive_demo())
