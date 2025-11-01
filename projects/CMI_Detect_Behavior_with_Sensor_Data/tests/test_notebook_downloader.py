"""Notebook downloaderのテストコード."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from codes.src.notebook_downloader.downloader import NotebookDownloader
from codes.src.notebook_downloader.scheduler import NotebookScheduler


class TestNotebookDownloader:
    """NotebookDownloaderのテストクラス."""

    def test_init(self):
        """初期化のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)
            # 独立した履歴ファイルパスを設定
            downloader.history_file = Path(temp_dir) / "history.json"

            assert downloader.competition == "test-competition"
            assert downloader.output_dir == Path(temp_dir)
            assert downloader.output_dir.exists()
            assert isinstance(downloader.download_history, dict)
            assert "downloads" in downloader.download_history
            assert "last_check" in downloader.download_history

    def test_load_history_new_file(self):
        """新規履歴ファイルのテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)
            # 独立した履歴ファイルパスを設定
            downloader.history_file = Path(temp_dir) / "history.json"

            # 履歴ファイルが存在しない場合のデフォルト値
            history = downloader.load_history()
            assert history == {"downloads": [], "last_check": None}

    def test_load_history_existing_file(self):
        """既存履歴ファイルのテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)

            # テスト用履歴データを作成
            test_history = {
                "downloads": [
                    {
                        "kernel_ref": "test/kernel",
                        "download_time": "2025-01-01T00:00:00",
                        "title": "Test Kernel",
                        "author": "TestUser",
                        "votes": 100,
                        "file_path": "/test/path",
                    }
                ],
                "last_check": "2025-01-01T00:00:00",
            }

            # 履歴ファイルを作成
            downloader.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(downloader.history_file, "w", encoding="utf-8") as f:
                json.dump(test_history, f)

            # 履歴を再読み込み
            history = downloader.load_history()
            assert history == test_history

    def test_save_history(self):
        """履歴保存のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)

            # テストデータを追加
            test_data = {
                "kernel_ref": "test/kernel",
                "download_time": "2025-01-01T00:00:00",
                "title": "Test Kernel",
                "author": "TestUser",
                "votes": 100,
                "file_path": "/test/path",
            }

            downloader.download_history["downloads"].append(test_data)
            downloader.save_history()

            # ファイルが作成されているかチェック
            assert downloader.history_file.exists()

            # 内容をチェック
            with open(downloader.history_file, encoding="utf-8") as f:
                saved_history = json.load(f)
                assert test_data in saved_history["downloads"]

    def test_is_already_downloaded(self):
        """ダウンロード済みチェックのテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)

            # テストデータを追加
            downloader.download_history["downloads"] = [
                {
                    "kernel_ref": "existing/kernel",
                    "download_time": "2025-01-01T00:00:00",
                    "title": "Existing Kernel",
                    "author": "TestUser",
                    "votes": 100,
                    "file_path": "/test/path",
                }
            ]

            # 既存kernelのテスト
            assert downloader.is_already_downloaded("existing/kernel") is True

            # 未存在kernelのテスト
            assert downloader.is_already_downloaded("new/kernel") is False

    @patch("codes.src.notebook_downloader.downloader.subprocess.run")
    def test_get_top_notebook_success(self, mock_run):
        """上位notebook取得成功のテスト."""
        # モックの戻り値を設定（実際のkaggle outputに合わせて調整）
        mock_result = MagicMock()
        mock_result.stdout = (
            "ref title author lastRunTime totalVotes\n"
            "test/kernel TestNotebook TestUser 2025-01-01 00:00:00 150\n"
        )
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)
            # 独立した履歴ファイルパスを設定
            downloader.history_file = Path(temp_dir) / "history.json"

            notebook_info = downloader.get_top_notebook()

            assert notebook_info is not None
            assert notebook_info["kernel_ref"] == "test/kernel"
            assert notebook_info["author"] == "TestUser"

    @patch("codes.src.notebook_downloader.downloader.subprocess.run")
    def test_get_top_notebook_failure(self, mock_run):
        """上位notebook取得失敗のテスト."""
        # エラーを発生させる
        mock_run.side_effect = Exception("API Error")

        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)

            notebook_info = downloader.get_top_notebook()
            assert notebook_info is None

    @patch("codes.src.notebook_downloader.downloader.subprocess.run")
    def test_download_notebook_success(self, mock_run):
        """notebookダウンロード成功のテスト."""
        # モックの戻り値を設定
        mock_result = MagicMock()
        mock_result.stdout = "Source code downloaded to /test/path"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)
            # 独立した履歴ファイルパスを設定
            downloader.history_file = Path(temp_dir) / "history.json"
            downloader.download_history = {"downloads": [], "last_check": None}

            kernel_info = {"kernel_ref": "test/kernel", "title": "Test Notebook", "author": "TestUser", "votes": 150}

            success = downloader.download_notebook(kernel_info)

            assert success is True
            assert len(downloader.download_history["downloads"]) == 1
            assert downloader.download_history["downloads"][0]["kernel_ref"] == "test/kernel"


class TestNotebookScheduler:
    """NotebookSchedulerのテストクラス."""

    def test_init(self):
        """初期化のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)
            scheduler = NotebookScheduler(downloader)

            assert scheduler.downloader == downloader
            assert scheduler.running is True

    def test_calculate_next_run(self):
        """次回実行時間計算のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)
            scheduler = NotebookScheduler(downloader)

            next_run = scheduler.calculate_next_run()

            # 10分(600秒) + 1-59秒の範囲内かチェック
            assert 601 <= next_run <= 659

    def test_handle_signal(self):
        """シグナル処理のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)
            scheduler = NotebookScheduler(downloader)

            # 初期状態
            assert scheduler.running is True

            # シグナル処理
            scheduler.handle_signal(2, None)  # SIGINT
            assert scheduler.running is False


@pytest.fixture
def temp_downloader():
    """テスト用ダウンローダーのフィクスチャ."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)


def test_integration_run_once():
    """統合テスト: 一度だけの実行."""
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = NotebookDownloader(competition_name="test-competition", output_dir=temp_dir)

        # get_top_notebookをモック
        with patch.object(downloader, "get_top_notebook") as mock_get:
            mock_get.return_value = None  # notebook取得失敗をシミュレート

            # 例外が発生しないことを確認
            downloader.run()
