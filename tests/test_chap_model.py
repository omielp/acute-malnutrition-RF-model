import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

from chap_model import normalize_dataframe, predict_model, train_model


class CausalPreprocessingTests(unittest.TestCase):
    def test_future_covariate_values_do_not_change_earlier_normalized_rows(self):
        historic = pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-02", "2020-03", "2020-04"],
                "location": ["A"] * 4,
                "mean_temperature": [20.0, None, 22.0, None],
                "population_u5": [1000.0, None, None, None],
            }
        )
        future = pd.DataFrame(
            {
                "time_period": ["2020-05"],
                "location": ["A"],
                "mean_temperature": [9999.0],
                "population_u5": [999999.0],
            }
        )

        baseline = normalize_dataframe(historic, require_target=False)
        combined = normalize_dataframe(pd.concat([historic, future], ignore_index=True), require_target=False)
        earlier_combined = combined[combined["time_period"].isin(historic["time_period"])].reset_index(drop=True)

        assert_frame_equal(baseline, earlier_combined)

    def test_chap_entrypoints_accept_optional_covariates(self):
        train = pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05"],
                "location": ["A"] * 5,
                "disease_cases": [10, 12, 11, 15, 14],
                "mean_temperature": [20.0, 21.0, 20.0, 22.0, 21.0],
            }
        )
        future = pd.DataFrame({"time_period": ["2020-06"], "location": ["A"]})

        with TemporaryDirectory() as directory:
            workdir = Path(directory)
            train_path = workdir / "train.csv"
            future_path = workdir / "future.csv"
            model_path = workdir / "model.pkl"
            output_path = workdir / "predictions.csv"
            train.to_csv(train_path, index=False)
            train.to_csv(workdir / "historic.csv", index=False)
            future.to_csv(future_path, index=False)

            train_model(str(train_path), str(model_path))
            predict_model(str(model_path), str(workdir / "historic.csv"), str(future_path), str(output_path))

            predictions = pd.read_csv(output_path)
            expected_columns = ["time_period", "location"] + [f"sample_{index}" for index in range(100)]
            self.assertEqual(predictions.columns.tolist(), expected_columns)
            self.assertEqual(len(predictions), 1)
            self.assertTrue(predictions.filter(like="sample_").ge(0).all().all())


if __name__ == "__main__":
    unittest.main()
