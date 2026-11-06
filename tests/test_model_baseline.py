import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model_baseline import evaluate, load_pair


ROOT = Path(__file__).resolve().parents[1]


class ModelBaselineTests(unittest.TestCase):
    def test_existing_train_test_files(self):
        train, test = load_pair(ROOT / "training_data.csv", ROOT / "test_data.csv")
        result = evaluate(train, test)
        self.assertEqual((result["train_rows"], result["test_rows"]), (250, 50))
        self.assertEqual(result, evaluate(train, test))
        self.assertGreaterEqual(result["selected_predictors"], 0)
        self.assertLessEqual(result["selected_predictors"], 50)

    def test_column_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.csv"
            test_path = Path(directory) / "test.csv"
            pd.DataFrame({"y": [1, 2], "x": [3, 4]}).to_csv(train_path, index=False)
            pd.DataFrame({"x": [3, 4], "y": [1, 2]}).to_csv(test_path, index=False)
            with self.assertRaisesRegex(ValueError, "columns must match"):
                load_pair(train_path, test_path)

    def test_missing_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.csv"
            test_path = Path(directory) / "test.csv"
            data = pd.DataFrame({"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, float("nan")]})
            data.to_csv(train_path, index=False)
            data.to_csv(test_path, index=False)
            with self.assertRaisesRegex(ValueError, "no missing"):
                load_pair(train_path, test_path)


if __name__ == "__main__":
    unittest.main()
