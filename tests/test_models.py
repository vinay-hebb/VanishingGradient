import pytest

torch = pytest.importorskip("torch")
from models import (
    VanillaDeepNetwork,
    ReLUNetwork,
    BatchNormNetwork,
    ResNetNetwork,
    get_model,
)

class TestVanillaDeepNetwork:
    def test_initialization(self):
        model = VanillaDeepNetwork(num_layers=5, input_size=784, hidden_size=128, output_size=10)
        assert len(model.layers) == 5
        
    def test_forward_pass(self):
        model = VanillaDeepNetwork(num_layers=5)
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        assert output.shape == (32, 10)
        
    def test_different_layer_counts(self):
        for num_layers in [3, 10, 20]:
            model = VanillaDeepNetwork(num_layers=num_layers)
            assert len(model.layers) == num_layers

class TestReLUNetwork:
    def test_initialization(self):
        model = ReLUNetwork(num_layers=5, input_size=784, hidden_size=128, output_size=10)
        assert len(model.layers) == 5
        
    def test_forward_pass(self):
        model = ReLUNetwork(num_layers=5)
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        assert output.shape == (32, 10)
        
    def test_gradient_flow(self):
        model = ReLUNetwork(num_layers=5)
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

class TestBatchNormNetwork:
    def test_initialization(self):
        model = BatchNormNetwork(num_layers=5, input_size=784, hidden_size=128, output_size=10)
        assert len(model.layers) == 5
        assert len(model.batch_norms) == 3
        
    def test_forward_pass(self):
        model = BatchNormNetwork(num_layers=5)
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        assert output.shape == (32, 10)
        
    def test_batch_normalization_effect(self):
        model = BatchNormNetwork(num_layers=5)
        model.eval()
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        assert not torch.isnan(output).any()

class TestResNetNetwork:
    def test_initialization(self):
        model = ResNetNetwork(num_layers=10, input_size=784, hidden_size=128, output_size=10)
        assert len(model.res_blocks) > 0
        
    def test_forward_pass(self):
        model = ResNetNetwork(num_layers=10)
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        assert output.shape == (32, 10)
        
    def test_residual_connections(self):
        model = ResNetNetwork(num_layers=10)
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert param.grad is not None

class TestGetModel:
    def test_get_vanilla_model(self):
        model = get_model('vanilla', num_layers=5)
        assert isinstance(model, VanillaDeepNetwork)
        
    def test_get_relu_model(self):
        model = get_model('relu', num_layers=5)
        assert isinstance(model, ReLUNetwork)
        
    def test_get_batchnorm_model(self):
        model = get_model('batchnorm', num_layers=5)
        assert isinstance(model, BatchNormNetwork)
        
    def test_get_resnet_model(self):
        model = get_model('resnet', num_layers=10)
        assert isinstance(model, ResNetNetwork)
        
    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            get_model('invalid_type', num_layers=5)
            
    def test_model_with_custom_parameters(self):
        model = get_model('relu', num_layers=7, input_size=1000, hidden_size=256, output_size=20)
        input_tensor = torch.randn(16, 1000)
        output = model(input_tensor)
        assert output.shape == (16, 20)
