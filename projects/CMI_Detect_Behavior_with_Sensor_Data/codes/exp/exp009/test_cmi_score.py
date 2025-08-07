"""CMI Score公式実装のテストコード."""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model import compute_cmi_score


def test_compute_cmi_score():
    """compute_cmi_score関数のテスト."""

    # Kaggle公式の target_gestures と non_target_gestures
    target_gestures = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Neck - pinch skin",
        "Neck - scratch",
    ]

    non_target_gestures = [
        "Write name on leg",
        "Wave hello",
        "Glasses on/off",
        "Text on phone",
        "Write name in air",
        "Feel around in tray and pull out an object",
        "Scratch knee/leg skin",
        "Pull air toward your face",
        "Drink from bottle/cup",
        "Pinch knee/leg skin",
    ]

    print("Testing compute_cmi_score function...")
    print("=" * 50)

    # テストケース1: 完全一致
    gesture_true = ["Above ear - pull hair", "Text on phone", "Eyebrow - pull hair", "Wave hello"]
    gesture_pred = ["Above ear - pull hair", "Text on phone", "Eyebrow - pull hair", "Wave hello"]

    score1 = compute_cmi_score(gesture_true, gesture_pred, target_gestures, non_target_gestures)
    print(f"Test 1 (Perfect match): {score1:.4f} (Expected: 1.0000)")

    # テストケース2: バイナリは完全一致、マルチクラスでターゲット間違い
    gesture_true = ["Above ear - pull hair", "Text on phone", "Eyebrow - pull hair", "Wave hello"]
    gesture_pred = ["Forehead - scratch", "Text on phone", "Neck - scratch", "Wave hello"]

    score2 = compute_cmi_score(gesture_true, gesture_pred, target_gestures, non_target_gestures)
    print(f"Test 2 (Binary match, multiclass error): {score2:.4f}")

    # テストケース3: 全て不一致
    gesture_true = ["Above ear - pull hair", "Text on phone", "Eyebrow - pull hair", "Wave hello"]
    gesture_pred = ["Text on phone", "Above ear - pull hair", "Wave hello", "Eyebrow - pull hair"]

    score3 = compute_cmi_score(gesture_true, gesture_pred, target_gestures, non_target_gestures)
    print(f"Test 3 (All mismatch): {score3:.4f} (Expected: 0.0000)")

    # テストケース4: Kaggleの公式サンプルデータと同等
    gesture_true = ["Eyebrow - pull hair"] * 4
    gesture_pred = ["Forehead - pull hairline"] * 4

    score4 = compute_cmi_score(gesture_true, gesture_pred, target_gestures, non_target_gestures)
    print(f"Test 4 (Kaggle official example): {score4:.4f} (Expected: 0.5000)")

    # テストケース5: non-target同士（same non-target）
    gesture_true = ["Text on phone"] * 4
    gesture_pred = ["Text on phone"] * 4

    score5 = compute_cmi_score(gesture_true, gesture_pred, target_gestures, non_target_gestures)
    print(f"Test 5 (Non-target match): {score5:.4f}")

    # テストケース6: non-target間違い（different non-targets）
    gesture_true = ["Text on phone"] * 4
    gesture_pred = ["Wave hello"] * 4

    score6 = compute_cmi_score(gesture_true, gesture_pred, target_gestures, non_target_gestures)
    print(f"Test 6 (Non-target mismatch): {score6:.4f}")

    # テストケース7: Kaggle公式docstringの例
    gesture_true = ["Eyebrow - pull hair"] * 4  # Target
    gesture_pred = ["Text on phone"] * 4  # Non-target

    score7 = compute_cmi_score(gesture_true, gesture_pred, target_gestures, non_target_gestures)
    print(f"Test 7 (Kaggle docstring example): {score7:.4f} (Expected: 0.0000)")

    print("=" * 50)
    print("CMI Score function test completed!")

    # 基本的な妥当性チェック
    assert 0.0 <= score1 <= 1.0, f"Score should be in [0,1], got {score1}"
    assert 0.0 <= score2 <= 1.0, f"Score should be in [0,1], got {score2}"
    assert 0.0 <= score3 <= 1.0, f"Score should be in [0,1], got {score3}"
    assert 0.0 <= score4 <= 1.0, f"Score should be in [0,1], got {score4}"
    assert 0.0 <= score5 <= 1.0, f"Score should be in [0,1], got {score5}"
    assert 0.0 <= score6 <= 1.0, f"Score should be in [0,1], got {score6}"
    assert 0.0 <= score7 <= 1.0, f"Score should be in [0,1], got {score7}"

    # 期待される値のチェック
    assert abs(score1 - 1.0) < 1e-6, f"Perfect match should be 1.0, got {score1}"
    assert abs(score4 - 0.5) < 1e-6, f"Kaggle official example should be 0.5, got {score4}"
    assert abs(score7 - 0.0) < 1e-6, f"Target vs Non-target should be 0.0, got {score7}"

    print("✓ All validation tests passed!")

    return True


if __name__ == "__main__":
    test_compute_cmi_score()
