from fastapi.testclient import TestClient
import time

from crowdstop.models.api import CameraUpdateRequest

from .conftest import client

def test_docs(client: TestClient):
    response = client.get('/docs')
    assert response.status_code == 200
    for text in ['DOCTYPE html', 'swagger-ui']:
        assert text in response.text

def test_openapi_json(client: TestClient):
    response = client.get('/openapi.json')
    assert response.status_code == 200
    openapi_spec = response.json()
    assert openapi_spec['openapi'].startswith('3.'), \
        f'Expected Open API spec 3+, got {openapi_spec["openapi"]}'
    
def test_health(client: TestClient):
    response = client.get('/health').json()
    assert response.get('status') == 'healthy', 'Expected "status" == "healthy" in response'

def test_update(client: TestClient):
    response = client.put(
        '/camera/12345', 
        json=CameraUpdateRequest(timestamp=int(time.time()), density=10).model_dump()
    )
    assert response.status_code == 200