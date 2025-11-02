import pandas as pd


def make_submission(test: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    try:
        assert len(submission) == len(test)
    except AssertionError:
        print(f"len(sample submission) = {len(submission)}, len(test) = {len(test)}")
        raise AssertionError

    submission["prediction"] = test

    return submission
