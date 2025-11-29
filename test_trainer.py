import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from trainer import (
    GradientTracker,
    create_synthetic_dataset,
    train_model,
    export_model_to_onnx,
    WANDB_AVAILABLE,
    build_export_subdir,
    create_results_entry,
    write_results_file,
)
from models import get_model

class TestGradientTracker:
    def test_initialization(self):
        tracker = GradientTracker()
        assert tracker.gradients_history == []
        assert tracker.loss_history == []
        assert tracker.epoch_gradients == {}
        
    def test_capture_gradients(self):
        tracker = GradientTracker()
        model = get_model('relu', num_layers=5)
        
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        gradients, layer_names = tracker.capture_gradients(model, epoch=0)
        
        assert len(gradients) > 0
        assert len(layer_names) > 0
        assert len(gradients) == len(layer_names)
        assert 0 in tracker.epoch_gradients
        
    def test_add_loss(self):
        tracker = GradientTracker()
        tracker.add_loss(0.5)
        tracker.add_loss(0.3)
        tracker.add_loss(0.1)
        
        assert len(tracker.loss_history) == 3
        assert tracker.loss_history == [0.5, 0.3, 0.1]
        
    def test_get_gradient_heatmap_data(self):
        tracker = GradientTracker()
        model = get_model('relu', num_layers=5)
        
        for epoch in range(3):
            input_tensor = torch.randn(32, 784)
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
            tracker.capture_gradients(model, epoch=epoch)
        
        heatmap_data = tracker.get_gradient_heatmap_data()
        
        assert heatmap_data is not None
        assert 'data' in heatmap_data
        assert 'epochs' in heatmap_data
        assert 'layer_names' in heatmap_data
        assert len(heatmap_data['epochs']) == 3
        
    def test_get_gradient_heatmap_data_empty(self):
        tracker = GradientTracker()
        heatmap_data = tracker.get_gradient_heatmap_data()
        assert heatmap_data is None

class TestCreateSyntheticDataset:
    def test_default_parameters(self):
        X, y = create_synthetic_dataset()
        assert X.shape == (200, 784)
        assert y.shape == (200,)
        assert y.dtype == torch.long
        
    def test_custom_parameters(self):
        X, y = create_synthetic_dataset(num_samples=500, input_size=100, output_size=5)
        assert X.shape == (500, 100)
        assert y.shape == (500,)
        assert y.max() < 5
        assert y.min() >= 0

class TestTrainModel:
    def test_train_vanilla_model(self):
        result = train_model(model_type='vanilla', num_layers=5, num_epochs=2, learning_rate=0.01)
        
        assert 'tracker' in result
        assert 'model' in result
        assert 'final_loss' in result
        assert 'wandb_run' in result
        assert 'wandb_url' in result
        assert len(result['tracker'].loss_history) == 2
        
    def test_train_relu_model(self):
        result = train_model(model_type='relu', num_layers=5, num_epochs=2, learning_rate=0.01)
        
        assert 'tracker' in result
        assert 'model' in result
        assert isinstance(result['final_loss'], float)
        assert 'wandb_run' in result
        assert 'wandb_url' in result
        
    def test_train_batchnorm_model(self):
        result = train_model(model_type='batchnorm', num_layers=5, num_epochs=2, learning_rate=0.01)
        
        assert 'tracker' in result
        assert len(result['tracker'].loss_history) > 0
        assert 'wandb_url' in result
        
    def test_train_resnet_model(self):
        result = train_model(model_type='resnet', num_layers=10, num_epochs=2, learning_rate=0.01)
        
        assert 'tracker' in result
        assert 'model' in result
        assert 'wandb_run' in result
        assert 'wandb_url' in result
        
    def test_gradient_capture_during_training(self):
        result = train_model(model_type='relu', num_layers=5, num_epochs=3, learning_rate=0.01)
        tracker = result['tracker']
        
        assert len(tracker.epoch_gradients) == 3
        for epoch in range(3):
            assert epoch in tracker.epoch_gradients
            
    def test_loss_decreases(self):
        result = train_model(model_type='relu', num_layers=5, num_epochs=10, learning_rate=0.01)
        loss_history = result['tracker'].loss_history
        
        assert len(loss_history) == 10
        assert loss_history[-1] <= loss_history[0]

    def test_wandb_logging_creates_offline_run(self, tmp_path):
        if not WANDB_AVAILABLE:
            pytest.skip('wandb package not available')
        run_dir = tmp_path / 'wandb' / 'vanilla'
        result = train_model(
            model_type='vanilla',
            num_layers=3,
            num_epochs=1,
            learning_rate=0.01,
            wandb_run_dir=str(run_dir)
        )
        assert result['wandb_run']
        assert 'wandb_url' in result
        run_path = Path(result['wandb_run'])
        assert run_path.exists()
        history_files = list(run_path.rglob('wandb-history.jsonl'))
        summary_files = list(run_path.rglob('wandb-summary.json'))
        assert history_files or summary_files, "Expected wandb offline files to be created"

class TestExportModelToONNX:
    def test_export_model(self):
        model = get_model('relu', num_layers=5)
        filename = 'test_model.onnx'
        
        result_filename = export_model_to_onnx(model, filename)
        
        assert result_filename == filename
        assert os.path.exists(filename)
        
        if os.path.exists(filename):
            os.remove(filename)
            
    def test_export_different_models(self):
        for model_type in ['vanilla', 'relu', 'batchnorm', 'resnet']:
            model = get_model(model_type, num_layers=5)
            filename = f'test_{model_type}.onnx'
            
            export_model_to_onnx(model, filename)
            assert os.path.exists(filename)
            
            if os.path.exists(filename):
                os.remove(filename)


class TestResultsPersistence:
    def test_results_file_written(self, tmp_path):
        cache_key = build_export_subdir(2, 1, 0.01, 4)
        export_dir = tmp_path / cache_key
        export_dir.mkdir(parents=True, exist_ok=True)

        results_payload = {}
        for model_type in ['vanilla', 'relu']:
            result = train_model(
                model_type=model_type,
                num_layers=2,
                num_epochs=1,
                learning_rate=0.01,
                hidden_size=4,
                wandb_run_dir='',  # disable wandb writes during the test
            )
            onnx_path = export_model_to_onnx(
                result['model'], str(export_dir / f'{model_type}.onnx'))
            results_payload[model_type] = create_results_entry(
                model_type=model_type,
                tracker=result['tracker'],
                onnx_path=onnx_path,
                wandb_run=result.get('wandb_run'),
                wandb_url=result.get('wandb_url'),
                cache_key=cache_key,
            )

        results_file = write_results_file(str(export_dir), results_payload)
        assert os.path.exists(results_file)

        saved = json.loads((tmp_path / cache_key / 'results.json').read_text())
        assert set(saved.keys()) == set(results_payload.keys())
        for entry in saved.values():
            assert entry['model_file'].endswith('.onnx')
            assert len(entry['loss_history']) == 1
            assert entry['heatmap_data'] is not None
