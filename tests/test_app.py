import math
import unittest
from unittest.mock import patch

from pydantic import ValidationError

import app


class FakeModel:
    def predict(self, rows):
        return [1]


def valid_transaction():
    return {
        "Time": 136131.0,
        "V_features": [0.0] * 28,
        "Amount": 0.9,
    }


class TransactionValidationTests(unittest.TestCase):
    def test_valid_prediction_preserves_response_shape(self):
        with patch.object(app, "model", FakeModel()):
            response = app.predict(app.Transaction(**valid_transaction()))

        self.assertEqual(response, {"is_fraud": True})

    def test_rejects_wrong_feature_count(self):
        for feature_count in (27, 29):
            with self.subTest(feature_count=feature_count):
                transaction = valid_transaction()
                transaction["V_features"] = [0.0] * feature_count
                with self.assertRaises(ValidationError):
                    app.Transaction(**transaction)

    def test_rejects_non_finite_values(self):
        for invalid_value in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid_value=invalid_value):
                transaction = valid_transaction()
                transaction["V_features"][0] = invalid_value
                with self.assertRaises(ValidationError):
                    app.Transaction(**transaction)

        for field in ("Time", "Amount"):
            for invalid_value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, invalid_value=invalid_value):
                    transaction = valid_transaction()
                    transaction[field] = invalid_value
                    with self.assertRaises(ValidationError):
                        app.Transaction(**transaction)

    def test_health_endpoint_reports_live_process(self):
        self.assertEqual(app.health(), {"status": "ok"})

    def test_readiness_reports_loaded_model(self):
        with patch.object(app, "model", FakeModel()):
            self.assertEqual(app.ready(), {"status": "ready"})

    def test_readiness_and_prediction_report_unavailable_model(self):
        with patch.object(app, "model", None):
            readiness = app.ready()
            prediction = app.predict(app.Transaction(**valid_transaction()))

        self.assertEqual(readiness.status_code, 503)
        self.assertEqual(prediction.status_code, 503)


if __name__ == "__main__":
    unittest.main()
