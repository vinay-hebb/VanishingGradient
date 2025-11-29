import argparse
import json
import logging
import os
import random
import shutil
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import get_model


def _ensure_distutils_shim() -> None:
    """
    wandb (via dockerpycreds) imports distutils.spawn/util, which are removed in Python 3.12+.
    Provide a minimal shim using shutil.which and a copy of strtobool so wandb can import.
    """
    if 'distutils.spawn' in sys.modules and 'distutils.util' in sys.modules:
        return
    try:
        import distutils.spawn  # type: ignore
        import distutils.util  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    distutils_mod = sys.modules.get('distutils') or types.ModuleType(
        'distutils')
    spawn_mod = types.ModuleType('distutils.spawn')
    spawn_mod.find_executable = shutil.which

    util_mod = types.ModuleType('distutils.util')

    def _strtobool(val):
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return 1
        if val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        raise ValueError(f'invalid truth value {val}')

    util_mod.strtobool = _strtobool

    distutils_mod.spawn = spawn_mod
    distutils_mod.util = util_mod
    sys.modules['distutils'] = distutils_mod
    sys.modules['distutils.spawn'] = spawn_mod
    sys.modules['distutils.util'] = util_mod


_ensure_distutils_shim()

os.environ.setdefault('WANDB_DISABLE_SERVICE', 'true')
os.environ.setdefault('WANDB_MODE', 'offline')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - wandb is a runtime dependency
    wandb = None
    WANDB_AVAILABLE = False
    logging.getLogger(__name__).warning(
        'wandb import failed; disabling wandb integration', exc_info=exc)

WANDB_MODE = os.environ.get('WANDB_MODE', 'online')

DEFAULT_SEED = 42
AVAILABLE_MODEL_TYPES = ['vanilla', 'relu', 'batchnorm', 'resnet']


def format_learning_rate(value: float) -> str:
    formatted = format(value, '.8f').rstrip('0').rstrip('.')
    return formatted if formatted else '0'


def build_export_subdir(num_layers: int, num_epochs: int, learning_rate: float,
                        hidden_size: int) -> str:
    lr_str = format_learning_rate(learning_rate)
    return (f"layers_{num_layers}_epochs_{num_epochs}_lr_{lr_str}"
            f"_hidden_{hidden_size}")


def _log_wandb_issue(message: str, error: Exception | None = None) -> None:
    """Lightweight logger for wandb diagnostics."""
    logger = logging.getLogger(__name__)
    if error:
        logger.warning(message, exc_info=error)
    else:
        logger.warning(message)


def set_random_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python, numpy, and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GradientTracker:

    def __init__(self):
        self.gradients_history = []
        self.loss_history = []
        self.epoch_gradients = {}

    def capture_gradients(self, model, epoch):
        gradients = []
        layer_names = []

        # Prefer grabbing gradients from linear layers in module order so models with
        # different attribute names (layers vs res_blocks) align.
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                grad_norm = module.weight.grad.abs().mean().item()
                gradients.append(grad_norm)
                layer_names.append(
                    f"{module_name}.weight" if module_name else "weight")

        if not gradients:
            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grad_norm = param.grad.abs().mean().item()
                    gradients.append(grad_norm)
                    layer_names.append(name)

        self.epoch_gradients[epoch] = {
            'gradients': gradients,
            'layer_names': layer_names
        }

        return gradients, layer_names

    def add_loss(self, loss):
        self.loss_history.append(loss)

    def get_gradient_heatmap_data(self):
        if not self.epoch_gradients:
            return None

        epochs = sorted(self.epoch_gradients.keys())
        num_layers = len(self.epoch_gradients[epochs[0]]['gradients'])

        heatmap_data = []
        for layer_idx in range(num_layers):
            layer_grads = []
            for epoch in epochs:
                layer_grads.append(
                    self.epoch_gradients[epoch]['gradients'][layer_idx])
            heatmap_data.append(layer_grads)

        return {
            'data': heatmap_data,
            'epochs': epochs,
            'layer_names': self.epoch_gradients[epochs[0]]['layer_names']
        }


def _get_final_epoch_data(tracker: GradientTracker):
    if not tracker.epoch_gradients:
        return [], []
    final_epoch = max(tracker.epoch_gradients.keys())
    epoch_data = tracker.epoch_gradients.get(final_epoch, {})
    return epoch_data.get('gradients', []), epoch_data.get('layer_names', [])


def create_results_entry(model_type: str, tracker: GradientTracker,
                         onnx_path: str | None, wandb_run: str | None,
                         wandb_url: str | None, cache_key: str):
    final_layer_gradients, final_layer_names = _get_final_epoch_data(tracker)
    return {
        'loss_history': tracker.loss_history, 'final_gradients': final_layer_gradients,
        'layer_names': final_layer_names, 'heatmap_data': tracker.get_gradient_heatmap_data(),
        'model_file': os.path.basename(onnx_path) if onnx_path else None, 'wandb_run': wandb_run,
        'wandb_run_name': f'{cache_key}/{model_type}', 'wandb_url': wandb_url
    }


def write_results_file(export_dir: str, results: dict):
    os.makedirs(export_dir, exist_ok=True)
    results_path = os.path.join(export_dir, 'results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp)
    return results_path


def create_synthetic_dataset(
    num_samples=200,
    input_size=784,
    output_size=10,
    seed: int | None = DEFAULT_SEED,
):
    if seed is not None:
        set_random_seed(seed)
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples, ))
    return X, y


def _init_wandb_run(run_dir: str | None, run_name: str, config: dict):
    if not run_dir:
        return None
    if not WANDB_AVAILABLE:
        _log_wandb_issue(
            'wandb package is not available; run links will be disabled')
        return None
    try:
        os.makedirs(run_dir, exist_ok=True)
        settings = wandb.Settings(start_method='thread',
                                  init_timeout=30,
                                  _disable_service=True)
        requested_mode = WANDB_MODE
        mode_to_use = requested_mode
        api_key = os.environ.get('WANDB_API_KEY')
        if requested_mode != 'offline' and not api_key:
            mode_to_use = 'offline'
            _log_wandb_issue(
                'WANDB_API_KEY not detected; defaulting to offline logging')

        def start(mode: str):
            return wandb.init(
                project='Gradient',
                mode=mode,
                name=run_name,
                dir=run_dir,
                config=config,
                reinit=True,
                settings=settings,
            )

        try:
            return start(mode_to_use)
        except Exception as exc:
            _log_wandb_issue(
                f'Failed to start wandb in {mode_to_use} mode, retrying offline',
                exc)
            if mode_to_use != 'offline':
                try:
                    return start('offline')
                except Exception as offline_exc:
                    _log_wandb_issue(
                        'Failed to start wandb in offline mode; disabling run logging',
                        offline_exc)
                    return None
            return None
    except Exception as exc:
        _log_wandb_issue('Unexpected error while configuring wandb',
                         error=exc)
        return None


def _log_gradient_histograms_wandb(run, model, epoch, prefix):
    if run is None or wandb is None:
        return
    histogram_batch = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_tensor = param.grad.detach().cpu().view(-1).numpy()
        if grad_tensor.size == 0:
            continue
        tag = f'{prefix}/{name}'
        histogram_batch[tag] = wandb.Histogram(grad_tensor)
        if len(histogram_batch) >= 8:
            run.log(histogram_batch, step=epoch)
            histogram_batch = {}
    if histogram_batch:
        run.log(histogram_batch, step=epoch)


def train_model(
    model_type='vanilla',
    num_layers=10,
    num_epochs=20,
    learning_rate=0.01,
    hidden_size=128,
    seed: int | None = DEFAULT_SEED,
    wandb_run_dir: str | None = None,
):
    # num_layers = min(num_layers, 15)
    num_epochs = min(num_epochs, 30)

    if seed is not None:
        set_random_seed(seed)

    model = get_model(model_type, num_layers, hidden_size=hidden_size)

    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{model_type.upper()}] Model has {num_params:,} parameters (layers={num_layers}, hidden_size={hidden_size})"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    X_train, y_train = create_synthetic_dataset(
        num_samples=96, seed=None if seed is None else seed)

    tracker = GradientTracker()

    batch_size = 32
    num_batches = max(1, len(X_train) // batch_size)
    wandb_run = _init_wandb_run(
        wandb_run_dir,
        f'{model_type}-run',
        {
            'model_type': model_type,
            'num_layers': num_layers,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'hidden_size': hidden_size,
        },
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        tracker.add_loss(avg_loss)
        if wandb_run is not None:
            metrics = {f'{model_type}/loss': avg_loss}

        optimizer.zero_grad()
        outputs = model(X_train[:batch_size])
        loss = criterion(outputs, y_train[:batch_size])
        loss.backward()

        gradients, layer_names = tracker.capture_gradients(model, epoch)
        if wandb_run is not None:
            for grad_value, layer_name in zip(gradients, layer_names):
                metrics[f'{model_type}/gradients/{layer_name}'] = grad_value
            wandb_run.log(metrics, step=epoch)
            _log_gradient_histograms_wandb(
                wandb_run,
                model,
                epoch,
                f'{model_type}/gradient_hist',
            )

    wandb_run_path = None
    wandb_run_url = None
    if wandb_run is not None:
        wandb_run_path = wandb_run.dir
        wandb_run_url = getattr(wandb_run, 'url', None)
        wandb_run.finish()
    elif wandb_run_dir:
        wandb_run_path = wandb_run_dir

    return {
        'tracker': tracker,
        'model': model,
        'final_loss': tracker.loss_history[-1] if tracker.loss_history else 0,
        'wandb_run': wandb_run_path,
        'wandb_url': wandb_run_url
    }


def export_model_to_onnx(model, filename='model.onnx'):
    model.eval()
    dummy_input = torch.randn(1, 784)
    torch.onnx.export(model, (dummy_input, ),
                      filename,
                      training=torch.onnx.TrainingMode.EVAL,
                      input_names=['input'],
                      output_names=['output'],
                      # Use the legacy exporter to keep BatchNorm nodes from being
                      # folded into the adjacent linear layers for visualization.
                      dynamo=False)
    return filename


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Train Gradient demo models on synthetic data.')
    parser.add_argument(
        '--model-types',
        nargs='+',
        default=AVAILABLE_MODEL_TYPES,
        help=("Space-separated list of model types to train; defaults to all "
              "supported models. Use 'all' to run every architecture."),
    )
    parser.add_argument('--num-layers',
                        type=int,
                        default=3,
                        help='Number of layers (capped internally).')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=2,
                        help='Number of epochs to train (capped internally).')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.01,
                        help='Learning rate for SGD.')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=8,
                        help='Hidden layer width.')
    parser.add_argument('--seed',
                        type=int,
                        default=DEFAULT_SEED,
                        help='Random seed; set to -1 to disable seeding.')
    parser.add_argument('--wandb-run-dir',
                        type=str,
                        default='cache/wandb_runs/',
                        help='Directory to store wandb run data; set empty to disable.')
    parser.add_argument(
        '--export-onnx',
        type=str,
        default='cache/',
        help='Directory to export ONNX files (one per model type).')
    return parser.parse_args()


def _main():
    args = _parse_args()
    wandb_dir = args.wandb_run_dir or None
    seed = None if args.seed is not None and args.seed < 0 else args.seed

    cache_key = build_export_subdir(args.num_layers, args.num_epochs,
                                    args.learning_rate, args.hidden_size)
    onnx_export_dir = None
    if args.export_onnx:
        if os.path.exists(args.export_onnx) and not os.path.isdir(
                args.export_onnx):
            raise ValueError('--export-onnx must point to a directory')
        onnx_export_dir = os.path.join(args.export_onnx, cache_key)
        os.makedirs(onnx_export_dir, exist_ok=True)

    selected_models = args.model_types
    if 'all' in selected_models:
        selected_models = AVAILABLE_MODEL_TYPES
    # Remove duplicates while preserving order
    seen = set()
    ordered_models = []
    for mt in selected_models:
        if mt in seen:
            continue
        if mt not in AVAILABLE_MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {mt}")
        ordered_models.append(mt)
        seen.add(mt)

    summary = []
    results_payload = {}
    for model_type in ordered_models:
        print(f"=== Training {model_type} ===")
        result = train_model(
            model_type=model_type,
            num_layers=args.num_layers,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            seed=seed,
            wandb_run_dir=wandb_dir,
        )

        onnx_path = None
        if args.export_onnx:
            path = os.path.join(onnx_export_dir, f"{model_type}.onnx")
            onnx_path = export_model_to_onnx(result['model'], path)
            results_payload[model_type] = create_results_entry(
                model_type=model_type,
                tracker=result['tracker'],
                onnx_path=onnx_path,
                wandb_run=result.get('wandb_run'),
                wandb_url=result.get('wandb_url'),
                cache_key=cache_key,
            )

        summary.append({
            'model': model_type,
            'loss': result['final_loss'],
            'wandb_run': result.get('wandb_run'),
            'wandb_url': result.get('wandb_url'),
            'onnx': onnx_path,
        })

    print('--- Training complete ---')
    for entry in summary:
        print(f"[{entry['model']}] loss={entry['loss']:.6f}")
        if wandb_dir:
            print(f"  W&B dir: {entry['wandb_run']}")
            if entry.get('wandb_url'):
                print(f"  W&B url: {entry['wandb_url']}")
        if entry['onnx']:
            print(f"  ONNX: {entry['onnx']}")

    if onnx_export_dir and results_payload:
        results_file = write_results_file(onnx_export_dir, results_payload)
        print(f"Saved JSON results to {results_file}")


if __name__ == '__main__':
    _main()
