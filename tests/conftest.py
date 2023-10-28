import pytest
from fastapi.testclient import TestClient

from crowdstop.main_server import app

@pytest.fixture(scope='package')
def client() -> TestClient:
    with TestClient(app) as client:
        yield client