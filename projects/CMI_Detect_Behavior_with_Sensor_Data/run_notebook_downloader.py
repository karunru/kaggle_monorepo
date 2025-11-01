#!/usr/bin/env python
"""CMI Competition Top Notebook自動ダウンローダーのメイン実行スクリプト."""

import argparse
import sys
from pathlib import Path

from loguru import logger

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from codes.src.notebook_downloader.downloader import NotebookDownloader
from codes.src.notebook_downloader.scheduler import NotebookScheduler


def setup_logger(debug: bool = False) -> None:
    """ログ設定."""
    # 既存のハンドラーを削除
    logger.remove()

    # ログレベル設定
    level = "DEBUG" if debug else "INFO"

    # コンソール出力
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        colorize=True,
    )

    # ファイル出力
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "notebook_downloader.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="CMI Competition Top Notebook自動ダウンローダー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 永続的に実行
  python run_notebook_downloader.py
  
  # テストモード（3回実行）
  python run_notebook_downloader.py --test 3
  
  # 単発実行
  python run_notebook_downloader.py --once
  
  # デバッグモード
  python run_notebook_downloader.py --debug
  
  # カスタム設定
  python run_notebook_downloader.py --competition custom-comp --output ./custom_dir
        """,
    )

    parser.add_argument(
        "--competition",
        default="cmi-detect-behavior-with-sensor-data",
        help="コンペティション名 (default: cmi-detect-behavior-with-sensor-data)",
    )

    parser.add_argument(
        "--output", default="data/downloaded_notebooks", help="出力ディレクトリ (default: data/downloaded_notebooks)"
    )

    parser.add_argument("--once", action="store_true", help="一度だけ実行（スケジュール実行しない）")

    parser.add_argument("--test", type=int, metavar="N", help="テストモード: N回だけ実行")

    parser.add_argument("--debug", action="store_true", help="デバッグログを有効化")

    args = parser.parse_args()

    # ログ設定
    setup_logger(debug=args.debug)

    # 設定表示
    logger.info("=" * 60)
    logger.info("CMI Competition Top Notebook自動ダウンローダー")
    logger.info("=" * 60)
    logger.info(f"コンペティション: {args.competition}")
    logger.info(f"出力ディレクトリ: {args.output}")
    logger.info(f"実行モード: {'単発' if args.once else f'テスト({args.test}回)' if args.test else '永続'}")
    logger.info(f"デバッグモード: {'ON' if args.debug else 'OFF'}")
    logger.info("=" * 60)

    try:
        # ダウンローダー初期化
        downloader = NotebookDownloader(competition_name=args.competition, output_dir=args.output)

        if args.once:
            # 単発実行
            logger.info("単発実行モードで開始します")
            downloader.run()
            logger.info("単発実行完了")

        else:
            # スケジューラー初期化
            scheduler = NotebookScheduler(downloader)

            if args.test:
                # テストモード
                logger.info(f"テストモード: {args.test}回実行します")
                scheduler.run_with_count(args.test)
            else:
                # 永続実行
                logger.info("永続実行モードで開始します")
                logger.info("停止する場合は Ctrl+C を押してください")
                scheduler.run_forever()

    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        sys.exit(1)
    finally:
        logger.info("プログラムを終了します")


if __name__ == "__main__":
    main()
