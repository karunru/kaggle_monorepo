#!/usr/bin/env python3
"""EMAConfig実装のテストスクリプト."""

from config import Config, EMAConfig


def test_ema_config():
    """EMAConfigの動作テスト."""
    print("EMAConfig実装テスト開始")

    # 1. デフォルト設定テスト
    print("\n1. デフォルト設定テスト:")
    config = Config()
    print(f"  enabled: {config.ema.enabled}")
    print(f"  beta: {config.ema.beta}")
    print(f"  update_every: {config.ema.update_every}")
    print(f"  update_after_step: {config.ema.update_after_step}")

    # 2. カスタム設定テスト
    print("\n2. カスタム設定テスト:")
    custom_ema_config = EMAConfig(enabled=True, beta=0.999, update_every=2, update_after_step=50)
    config.ema = custom_ema_config
    print(f"  enabled: {config.ema.enabled}")
    print(f"  beta: {config.ema.beta}")
    print(f"  update_every: {config.ema.update_every}")
    print(f"  update_after_step: {config.ema.update_after_step}")

    # 3. EMA無効化テスト
    print("\n3. EMA無効化テスト:")
    config.ema.enabled = False
    print(f"  enabled: {config.ema.enabled}")

    # 4. 設定の辞書化テスト
    print("\n4. 設定の辞書化テスト:")
    config_dict = config.to_dict()
    print(f"  ema設定: {config_dict['ema']}")

    print("\n✅ すべてのテストが正常に完了しました!")


if __name__ == "__main__":
    test_ema_config()
