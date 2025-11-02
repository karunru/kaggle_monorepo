import argparse
from pathlib import Path


def get_latest_config_file(config_dir: Path = Path("codes/config")) -> Path | None:
    yaml_files = list(config_dir.glob("*.yml"))
    if not yaml_files:
        return None
    return max(yaml_files)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=get_latest_config_file(),
        help="Config file path",
        type=Path,
    )
    parser.add_argument("--log_dir", default="./log", help="Directory to save log")
    parser.add_argument("--debug", action="store_true", help="Whether to use debug mode")
    return parser


def get_preprocess_parser() -> argparse.ArgumentParser:
    parser = get_parser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing feature files")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Use subset of train.csv to calculate the feature",
    )
    return parser


def get_making_seed_average_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True, help="Base config file path")
    parser.add_argument("--num_seeds", required=True, help="num seeds")

    return parser


def get_seed_average_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Base output file path")

    return parser
