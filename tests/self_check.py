"""
启动前自检脚本。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig, PROJECT_CONFIG_PATH


def main() -> None:
    config = MemoryConfig()

    checks = [
        ("config_file", os.path.exists(PROJECT_CONFIG_PATH)),
        ("llm_provider", bool(config.llm_provider)),
        ("embedding_provider", bool(config.embedding_provider)),
        ("rerank_provider", bool(config.rerank_provider)),
        ("llm_api_key", bool(config.llm_api_key)),
        ("embedding_api_key", bool(config.embedding_api_key)),
        ("rerank_api_key", bool(config.rerank_api_key)),
        ("llm_model", bool(config.llm_model)),
        ("embedding_model", bool(config.embedding_model)),
        ("rerank_model", bool(config.rerank_model)),
        ("turns_before_extraction", config.conversation_turns_before_extraction >= 1),
    ]

    print("=== Memory System Self Check ===")
    print(f"config_file: {PROJECT_CONFIG_PATH}")
    print(f"llm_provider: {config.llm_provider}")
    print(f"embedding_provider: {config.embedding_provider}")
    print(f"rerank_provider: {config.rerank_provider}")

    failed = []
    for name, ok in checks:
        print(f"{name}: {'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)

    if failed:
        print("\nself-check failed:", ", ".join(failed))
        raise SystemExit(1)

    print("\nself-check passed")


if __name__ == "__main__":
    main()
