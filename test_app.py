import pytest
from app import app as flask_app

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_home_page(client):
    """
    GIVEN a Flask application
    WHEN the '/' page is requested (GET)
    THEN check that the response is valid
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b"Static Malware Detection" in response.data

def test_prediction_endpoint(client):
    """
    GIVEN a Flask application
    WHEN the '/predict' page is posted to with valid data
    THEN check that the response is valid and contains a prediction
    """
    # This is the same 22-feature list from your index.html
    # We create a dictionary to simulate the form data
    sample_data = {f'f{i+1}': 0 for i in range(22)} 
    
    response = client.post('/predict', data=sample_data)
    assert response.status_code == 200
    assert b"Prediction Result" in response.data
    assert b"The file is predicted to be:" in response.data