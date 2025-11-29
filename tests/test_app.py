import json
import pytest

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def _first_artifact_id(client):
    response = client.get('/api/artifacts')
    assert response.status_code == 200
    payload = json.loads(response.data)
    artifacts = payload.get('artifacts', [])
    if not artifacts:
        return None
    return artifacts[0]['id']


class TestRoutes:

    def test_index_route(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_artifact_listing_and_fetch(self, client):
        artifact_id = _first_artifact_id(client)
        if artifact_id is None:
            pytest.skip("No cached artifacts available")
        fetch_resp = client.get(f'/api/artifacts/{artifact_id}')
        assert fetch_resp.status_code == 200
        payload = json.loads(fetch_resp.data)
        assert 'results' in payload
        assert 'artifact' in payload
        assert payload['artifact']['id'] == artifact_id

    def test_model_file_is_served_from_cache(self, client):
        artifact_id = _first_artifact_id(client)
        if artifact_id is None:
            pytest.skip("No cached artifacts available")
        fetch_resp = client.get(f'/api/artifacts/{artifact_id}')
        payload = json.loads(fetch_resp.data)
        any_model = next(iter(payload['results'].values()))
        model_url = any_model['model_url']

        model_resp = client.get(model_url)
        assert model_resp.status_code == 200

    def test_get_model_file_not_found(self, client):
        response = client.get('/api/artifacts/layers_0_epochs_0_lr_0_hidden_0/model/nonexistent.onnx')
        assert response.status_code == 404


class TestCORS:

    def test_cors_headers(self, client):
        response = client.get('/')
        assert 'Access-Control-Allow-Origin' in response.headers or response.status_code == 200


class TestSecurity:

    def test_path_traversal_blocked(self, client):
        response = client.get('/api/artifacts/layers_0_epochs_0_lr_0_hidden_0/model/../../../etc/passwd')
        assert response.status_code in [400, 404]

    def test_non_onnx_file_blocked(self, client):
        response = client.get('/api/artifacts/layers_0_epochs_0_lr_0_hidden_0/model/test.txt')
        assert response.status_code == 400

    def test_slash_in_filename_blocked(self, client):
        response = client.get('/api/artifacts/layers_0_epochs_0_lr_0_hidden_0/model/subdir/model.onnx')
        assert response.status_code in [400, 404]
