import os
import pytest
from fastapi.testclient import TestClient

from crowdstop.main_server import app

os.environ['NEO4J_URL'] = 'bolt://neo4j:crowdstop@localhost:7687'

@pytest.fixture(scope='package')
def client() -> TestClient:
    with TestClient(app) as client:
        yield client