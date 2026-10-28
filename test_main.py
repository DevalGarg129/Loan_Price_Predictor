import unittest

from main import load_and_preprocess_data, train_and_evaluate


class LoanApprovalModelTests(unittest.TestCase):
    def test_load_and_preprocess_data_returns_features_and_target(self):
        X, y = load_and_preprocess_data()

        self.assertNotIn("Loan_Status", X.columns)
        self.assertEqual(len(X), len(y))
        self.assertTrue(set(y.unique()).issubset({0, 1}))

    def test_train_and_evaluate_returns_model_metrics(self):
        X, y = load_and_preprocess_data()
        results = train_and_evaluate(X, y)

        self.assertIn("RandomForest", results)
        self.assertIn("LogisticRegression", results)
        self.assertTrue(all(0 <= metrics["accuracy"] <= 1 for metrics in results.values()))


if __name__ == "__main__":
    unittest.main()
