import contextlib

import joblib
from tqdm.auto import tqdm


@contextlib.contextmanager
def tqdm_joblib(total: int | None = None, **kwargs):
    # with tqdm_joblib(total=len(unseen_cols_pairs)):
    #     result = joblib.Parallel(n_jobs=-1)(
    #         joblib.delayed(calc_corr)(feat_a_name, feat_b_name, 0.01) for feat_a_name, feat_b_name in unseen_cols_pairs
    #     )

    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()
