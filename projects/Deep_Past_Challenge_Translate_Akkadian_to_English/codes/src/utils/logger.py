import logging
import sys
from pathlib import Path


def configure_logger(config_name: Path, log_dir: Path | str, debug: bool):
    if isinstance(log_dir, str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = config_name.name.replace(".yml", ".log")
    log_filepath = log_dir / log_filename if isinstance(log_dir, Path) else Path(log_dir) / log_filename

    # delete the old log
    if log_filepath.exists():
        with open(log_filepath, mode="a"):
            pass

    level = logging.DEBUG if debug else logging.INFO

    # ルートロガーの設定
    logger = logging.getLogger()
    logger.setLevel(level)

    # フォーマッターの作成
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")

    # ファイルハンドラーの設定
    file_handler = logging.FileHandler(str(log_filepath))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 標準出力ハンドラーの設定
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.getLogger("matplotlib.font_manager").disabled = True


def create_logger(name: str, log_file: Path | str = None) -> logging.Logger:
    """
    ロガーを作成.

    Args:
        name: ロガー名
        log_file: ログファイルパス（オプション）

    Returns:
        設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # ハンドラーが既に存在する場合は削除（重複防止）
    if logger.handlers:
        logger.handlers.clear()

    # フォーマッターの作成
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")

    # 標準出力ハンドラー
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ファイルハンドラー（オプション）
    if log_file:
        if isinstance(log_file, str):
            log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
