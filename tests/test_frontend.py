import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestFrontend:
    def test_index_page_loads(self, client):
        response = client.get('/')
        assert response.status_code == 200
        assert b'Vanishing Gradient Visualizations' in response.data
        
    def test_index_contains_required_components(self, client):
        response = client.get('/')
        html = response.data.decode('utf-8')
        
        assert 'artifact-set' in html
        assert 'animate-btn' in html
        assert 'gradients-plot' in html
        assert 'heatmap-section' in html
        assert 'loss-section' in html
        assert 'wandb-section' in html
        
    def test_index_has_controls(self, client):
        response = client.get('/')
        html = response.data.decode('utf-8')
        
        assert 'artifact-set' in html
        assert 'animate-btn' in html
        assert 'train-btn' not in html
        assert 'num-layers' not in html
        assert 'num-epochs' not in html
        assert 'hidden-size' not in html
        
    def test_static_js_file_exists(self, client):
        response = client.get('/app.js')
        assert response.status_code in [200, 304]
        
    def test_chart_js_referenced(self, client):
        response = client.get('/')
        html = response.data.decode('utf-8')
        
        assert 'chart.js' in html.lower()
        
    def test_netron_referenced(self, client):
        response = client.get('/')
        html = response.data.decode('utf-8')
        
        assert 'netron' in html.lower()
