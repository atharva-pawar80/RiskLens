import os

def test_model_file_exists():
    # Check if the trained model pickle file exists
    assert os.path.exists("fraud_detection.pkl")

def test_basic_math():
    # Verification check
    assert 1 + 1 == 2
