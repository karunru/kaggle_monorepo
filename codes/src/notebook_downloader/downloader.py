"""Kaggle notebook自動ダウンロード機能."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from loguru import logger


class NotebookDownloader:
    """Kaggle notebookの自動ダウンロードクラス."""

    def __init__(self, competition_name: str, output_dir: str) -> None:
        """
        初期化.

        Args:
            competition_name: コンペティション名
            output_dir: ダウンロード先ディレクトリ
        """
        self.competition = competition_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = Path("codes/src/notebook_downloader/history.json")
        self.download_history = self.load_history()

        logger.info(f"NotebookDownloader initialized: {competition_name}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_history(self) -> dict:
        """ダウンロード履歴を読み込み."""
        if self.history_file.exists():
            try:
                with open(self.history_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("履歴ファイルの読み込みに失敗しました。新しい履歴を作成します。")

        return {"downloads": [], "last_check": None}

    def save_history(self) -> None:
        """ダウンロード履歴を保存."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.download_history, f, ensure_ascii=False, indent=2)
        logger.debug("履歴ファイルを保存しました")

    def get_top_notebook(self) -> dict[str, str] | None:
        """上位1つのnotebook情報を取得."""
        try:
            cmd = [
                "uv",
                "run",
                "kaggle",
                "kernels",
                "list",
                "--competition",
                self.competition,
                "--sort-by",
                "scoreDescending",
                "--page-size",
                "1",
            ]

            logger.debug(f"実行コマンド: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 出力をパースしてnotebook情報を取得
            lines = result.stdout.strip().split("\n")
            if len(lines) < 3:
                logger.warning("取得できたnotebook情報がありません")
                return None

            # ヘッダー行とセパレーター行をスキップして最初のデータ行を解析
            data_line = None
            for line in lines[1:]:  # ヘッダー行をスキップ
                if not line.startswith("-"):  # セパレーター行をスキップ
                    data_line = line.split()
                    break
            
            if not data_line or len(data_line) < 5:
                logger.warning("notebook情報の解析に失敗しました")
                return None

            kernel_ref = data_line[0]
            # format: ref title author lastRunTime(2 parts) totalVotes
            # 最後から: votes(-1), lastRunTime(-3,-2), author(-4)
            # テストデータに合わせて調整
            votes = int(data_line[-1])
            author = data_line[-4]  
            last_run_time = " ".join(data_line[-3:-1])
            title = " ".join(data_line[1:-4]) if len(data_line) > 4 else data_line[1]


            notebook_info = {
                "kernel_ref": kernel_ref,
                "title": title,
                "author": author,
                "last_run_time": last_run_time,
                "votes": votes,
            }

            logger.info(f"上位notebook取得: {kernel_ref} ({votes} votes)")
            return notebook_info

        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle API呼び出しエラー: {e}")
            logger.error(f"stderr: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return None

    def is_already_downloaded(self, kernel_ref: str) -> bool:
        """ダウンロード済みかチェック."""
        for download in self.download_history["downloads"]:
            if download["kernel_ref"] == kernel_ref:
                logger.debug(f"既にダウンロード済み: {kernel_ref}")
                return True
        return False

    def download_notebook(self, kernel_info: dict[str, str]) -> bool:
        """notebookをダウンロード."""
        kernel_ref = kernel_info["kernel_ref"]

        # ダウンロード先ディレクトリ名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_kernel_ref = kernel_ref.replace("/", "_")
        download_dir = self.output_dir / f"{safe_kernel_ref}_{timestamp}"

        try:
            cmd = [
                "uv",
                "run",
                "kaggle",
                "kernels",
                "pull",
                kernel_ref,
                "-p",
                str(download_dir),
                "-m",  # メタデータも取得
            ]

            logger.info(f"ダウンロード開始: {kernel_ref}")
            logger.debug(f"実行コマンド: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"ダウンロード完了: {download_dir}")
            logger.debug(f"stdout: {result.stdout}")

            # 履歴を更新
            download_record = {
                "kernel_ref": kernel_ref,
                "download_time": datetime.now().isoformat(),
                "title": kernel_info.get("title", ""),
                "author": kernel_info.get("author", ""),
                "votes": kernel_info.get("votes", 0),
                "file_path": str(download_dir),
            }
            self.download_history["downloads"].append(download_record)
            self.download_history["last_check"] = datetime.now().isoformat()
            self.save_history()

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"ダウンロードエラー: {e}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return False

    def run(self) -> None:
        """メイン処理."""
        logger.info("ダウンロード処理を開始します")

        # 1. 上位notebookを取得
        notebook_info = self.get_top_notebook()
        if not notebook_info:
            logger.warning("上位notebookの取得に失敗しました")
            return

        kernel_ref = notebook_info["kernel_ref"]

        # 2. ダウンロード済みかチェック
        if self.is_already_downloaded(kernel_ref):
            logger.info(f"スキップ: {kernel_ref} (既にダウンロード済み)")
            return

        # 3. ダウンロード実行
        success = self.download_notebook(notebook_info)
        if success:
            logger.info(f"新しいnotebookをダウンロードしました: {kernel_ref}")
        else:
            logger.error(f"ダウンロードに失敗しました: {kernel_ref}")


if __name__ == "__main__":
    # テスト実行用
    downloader = NotebookDownloader(
        competition_name="cmi-detect-behavior-with-sensor-data", output_dir="data/downloaded_notebooks"
    )
    downloader.run()
