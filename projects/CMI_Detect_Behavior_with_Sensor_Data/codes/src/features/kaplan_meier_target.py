import polars
from lifelines import KaplanMeierFitter
from src.features.base import Feature
from src.utils import timer


class KaplanMeierTarget(Feature):
    def create_features(
        self,
        train_df: polars.DataFrame,
        test_df: polars.DataFrame,
    ):
        with timer("load data"):
            train = train_df
            test = test_df

        with timer("Kaplan-Meier transform"):
            kmf = KaplanMeierFitter()
            kmf.fit(train.get_column("efs_time"), train.get_column("efs"))

        with timer("end"):
            self.train = train.with_columns(
                polars.Series(
                    name="kaplan_meier_target",
                    values=kmf.survival_function_at_times(train.get_column("efs_time")),
                ),
            ).select(["ID", "kaplan_meier_target"])

            self.test = test.select(["ID"])
