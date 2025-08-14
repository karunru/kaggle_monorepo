"""exp020からpywtを使った特徴量が削除されていることを確認するテスト."""

import ast
import sys
from pathlib import Path

import pytest


def test_no_pywt_import():
    """pywtがimportされていないことを確認."""
    dataset_path = Path(__file__).parent.parent / "codes" / "exp" / "exp020" / "dataset.py"

    # ファイルの内容を読み込み
    with open(dataset_path) as f:
        content = f.read()

    # pywtのimportがないことを確認
    assert "import pywt" not in content, "pywt import should be removed"
    assert "from pywt" not in content, "pywt import should be removed"


def test_no_wavelet_features_in_imu_cols():
    """imu_colsにウェーブレット特徴量が含まれていないことを確認."""
    dataset_path = Path(__file__).parent.parent / "codes" / "exp" / "exp020" / "dataset.py"
    
    # ファイルの内容を読み込んで、imu_cols定義を探す
    with open(dataset_path) as f:
        content = f.read()
    
    # ウェーブレット特徴量の名前パターン
    wavelet_patterns = [
        "wavelet_cA",
        "wavelet_cD1",
        "wavelet_cD2",
        "wavelet_cD3",
    ]
    
    # imu_colsの定義を探す（self.imu_cols = [ で始まる部分）
    import re
    imu_cols_match = re.search(r'self\.imu_cols = \[(.*?)\]', content, re.DOTALL)
    
    if imu_cols_match:
        imu_cols_str = imu_cols_match.group(1)
        
        # ウェーブレット特徴量が含まれていないことを確認
        for pattern in wavelet_patterns:
            assert pattern not in imu_cols_str, f"Wavelet feature pattern {pattern} should not be in imu_cols"


def test_no_add_wavelet_features_method():
    """_add_wavelet_featuresメソッドが存在しないことを確認."""
    dataset_path = Path(__file__).parent.parent / "codes" / "exp" / "exp020" / "dataset.py"

    # ASTを解析してメソッドの存在確認
    with open(dataset_path) as f:
        tree = ast.parse(f.read())

    # IMUDatasetクラスを探す
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "IMUDataset":
            # メソッド名を収集
            method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            assert "_add_wavelet_features" not in method_names, "_add_wavelet_features method should be removed"
            break


def test_no_wavelet_transform_call():
    """ウェーブレット変換の呼び出しがないことを確認."""
    dataset_path = Path(__file__).parent.parent / "codes" / "exp" / "exp020" / "dataset.py"

    with open(dataset_path) as f:
        content = f.read()

    # pywtの関数呼び出しがないことを確認
    assert "pywt.wavedec" not in content, "pywt.wavedec should not be called"
    assert "wavedec(" not in content, "wavedec should not be called"

    # _add_wavelet_featuresの呼び出しがないことを確認
    assert "_add_wavelet_features(" not in content, "_add_wavelet_features should not be called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
