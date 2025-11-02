import polars
import shirokumas
from sklearn.preprocessing import QuantileTransformer
from src.features.base import Feature
from src.utils import timer


class BasicFixLE(Feature):
    def create_features(
        self,
        train_df: polars.DataFrame,
        test_df: polars.DataFrame,
    ):
        with timer("load data"):
            train = train_df
            train_ids = train.get_column("ID").to_list()
            test = test_df

        with timer("concat train and test"):
            total = polars.concat([train, test], how="diagonal")

        with timer("label encoder"):
            not_cat_cols = ["ID", "donor_age", "age_at_hct", "efs", "efs_time"]
            label_encoder = shirokumas.OrdinalEncoder(cols=[col for col in total.columns if col not in not_cat_cols])
            label_encoder.fit(total)
            transformed_total = label_encoder.transform(total) + 2
            transformed_total = transformed_total.with_columns(
                [polars.col(col) % polars.col(col).n_unique() for col in transformed_total.columns],
            )
            total = total.with_columns(transformed_total)

        with timer("rankgauss transform"):
            transformer = QuantileTransformer(
                n_quantiles=1000,
                random_state=0,
                output_distribution="normal",
            ).set_output(transform="polars")
            total = total.with_columns(
                transformer.fit_transform(
                    total.select(["donor_age", "age_at_hct"]).fill_null(strategy="mean"),
                ),
            )

        with timer("end"):
            self.train = total.filter(polars.col("ID").is_in(train_ids)).select(polars.all().exclude("efs_time"))
            self.test = total.filter(~polars.col("ID").is_in(train_ids)).select(polars.all().exclude("efs_time"))
