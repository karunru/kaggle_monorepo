"""Notebook自動ダウンロードのスケジューリング機能."""

import random
import signal
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .downloader import NotebookDownloader


class NotebookScheduler:
    """Notebook自動ダウンロードのスケジューラー."""

    def __init__(self, downloader: "NotebookDownloader") -> None:
        """
        初期化.

        Args:
            downloader: NotebookDownloaderインスタンス
        """
        self.downloader = downloader
        self.running = True

        # シグナルハンドラーを設定
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        logger.info("NotebookScheduler初期化完了")

    def handle_signal(self, signum: int, frame) -> None:
        """
        シグナル処理（Ctrl+C等での安全な終了）.

        Args:
            signum: シグナル番号
            frame: フレームオブジェクト
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"停止シグナル({signal_name})を受信しました。安全に終了します...")
        self.running = False

    def calculate_next_run(self) -> int:
        """
        次回実行までの秒数を計算.

        Returns:
            5分 + ランダム1-59秒の合計秒数
        """
        base_seconds = 5 * 60  # 5分
        random_seconds = random.randint(1, 59)
        total_seconds = base_seconds + random_seconds

        logger.debug(f"次回実行まで: {total_seconds}秒 (5分 + {random_seconds}秒)")
        return total_seconds

    def run_once(self) -> bool:
        """
        一度だけダウンロード処理を実行.

        Returns:
            処理が成功したかどうか
        """
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{current_time}] ダウンロード処理開始")

            self.downloader.run()

            logger.info("ダウンロード処理完了")
            return True

        except Exception as e:
            logger.error(f"ダウンロード処理中にエラーが発生しました: {e}")
            return False

    def run_forever(self) -> None:
        """永続的にスケジュール実行."""
        logger.info("スケジューラーを開始します")
        logger.info("停止する場合は Ctrl+C を押してください")

        # 初回は即座に実行
        logger.info("初回ダウンロード処理を実行します")
        self.run_once()

        while self.running:
            try:
                # 次回実行時刻を計算
                wait_seconds = self.calculate_next_run()
                next_run = datetime.now() + timedelta(seconds=wait_seconds)

                logger.info(f"次回実行予定: {next_run.strftime('%Y-%m-%d %H:%M:%S')} ({wait_seconds}秒後)")

                # 待機（1秒単位でチェックして早期終了に対応）
                for _ in range(wait_seconds):
                    if not self.running:
                        logger.info("停止フラグが設定されました。待機を中断します")
                        break
                    time.sleep(1)

                if not self.running:
                    break

                # ダウンロード処理実行
                success = self.run_once()

                if not success:
                    logger.warning("ダウンロード処理に失敗しました。1分後にリトライします")
                    time.sleep(60)

            except KeyboardInterrupt:
                logger.info("KeyboardInterruptを受信しました")
                break
            except Exception as e:
                logger.error(f"スケジューラーでエラーが発生しました: {e}")
                logger.info("1分待機してから継続します")
                time.sleep(60)

        logger.info("スケジューラーを終了しました")

    def run_with_count(self, max_runs: int) -> None:
        """
        指定回数だけ実行（テスト用）.

        Args:
            max_runs: 最大実行回数
        """
        logger.info(f"テストモード: 最大{max_runs}回実行します")

        run_count = 0
        while self.running and run_count < max_runs:
            run_count += 1
            logger.info(f"実行回数: {run_count}/{max_runs}")

            success = self.run_once()

            if run_count < max_runs:  # 最後の実行でなければ待機
                if success:
                    wait_seconds = self.calculate_next_run()
                else:
                    wait_seconds = 60  # エラー時は1分待機

                next_run = datetime.now() + timedelta(seconds=wait_seconds)
                logger.info(f"次回実行: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

                for _ in range(wait_seconds):
                    if not self.running:
                        break
                    time.sleep(1)

        logger.info(f"テストモード完了: {run_count}回実行しました")


if __name__ == "__main__":
    # テスト実行用
    from .downloader import NotebookDownloader

    downloader = NotebookDownloader(
        competition_name="cmi-detect-behavior-with-sensor-data", output_dir="data/downloaded_notebooks"
    )

    scheduler = NotebookScheduler(downloader)

    # テストモード（3回実行）
    scheduler.run_with_count(3)
