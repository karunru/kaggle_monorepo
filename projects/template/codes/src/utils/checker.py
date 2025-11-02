import logging
from pathlib import Path


def feature_existence_checker(feature_path: Path, feature_names: list[str]) -> bool:
    features = [f.name for f in feature_path.glob("*.arrow")]
    for f in feature_names:
        if f + "_train.arrow" not in features:
            logging.debug(f"not exists {f}_train.arrow")
            return False
        if f + "_test.arrow" not in features:
            logging.debug(f"not exists {f}_train.arrow")
            return False
    return True
