#!/bin/bash

# exp002~exp005 シンプル実行スクリプト

# 設定
DRY_RUN=false
TARGET_EXP=""

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --exp)
            TARGET_EXP="$2"
            shift 2
            ;;
        --help)
            echo "使用法: $0 [--dry-run] [--exp EXPERIMENT]"
            echo "例:"
            echo "  $0                   # 全実験実行"
            echo "  $0 --dry-run         # ドライラン"
            echo "  $0 --exp exp005      # exp005のみ実行"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 対象実験の決定
if [ -n "$TARGET_EXP" ]; then
    EXPERIMENTS=("$TARGET_EXP")
else
    EXPERIMENTS=("exp002" "exp003" "exp004" "exp005" "exp006" "exp007")
fi

echo "=== exp002~exp006 実行スクリプト ==="
echo "対象実験: ${EXPERIMENTS[*]}"

if [ "$DRY_RUN" = true ]; then
    echo "ドライランモード（実際には実行されません）"
fi

echo ""

# 実験実行
for exp in "${EXPERIMENTS[@]}"; do
    echo "--- $exp ---"
    
    # ディレクトリ確認
    if [ ! -d "$exp" ]; then
        echo "エラー: ディレクトリ $exp が見つかりません"
        continue
    fi
    
    if [ ! -f "$exp/train.py" ]; then
        echo "エラー: $exp/train.py が見つかりません"
        continue
    fi
    
    # 実行コマンド
    cmd="cd $exp && uv run python train.py"
    echo "実行: $cmd"
    
    if [ "$DRY_RUN" = false ]; then
        echo "実行開始..."
        start_time=$(date +%s)
        
        # 実際に実行
        if (cd "$exp" && uv run python train.py); then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "$exp 完了 (${duration}秒)"
        else
            echo "$exp 失敗"
        fi
    else
        echo "ドライラン: 実際には実行されません"
    fi
    
    echo ""
done

echo "=== 実行完了 ==="