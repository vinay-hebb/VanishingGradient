import re
import json
import os
import logging
import socket
import threading
import sys
from urllib.parse import urljoin
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

try:
    import netron
    NETRON_AVAILABLE = True
except ImportError:
    netron = None
    NETRON_AVAILABLE = False

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Stream logs to stdout so they show up in local terminals and hosted consoles.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
if not any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, 'stream', None) is sys.stdout
        for h in root_logger.handlers):
    root_logger.addHandler(console_handler)
if not any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, 'stream', None) is sys.stdout
        for h in logging.getLogger('werkzeug').handlers):
    logging.getLogger('werkzeug').addHandler(console_handler)

# Use root handlers for the Flask app logger to avoid duplicate streams.
app.logger.handlers.clear()
app.logger.setLevel(logging.INFO)
app.logger.propagate = True

# Training is disabled in deployment; we only serve cached artifacts.
WANDB_AVAILABLE = True
WANDB_MODE = os.environ.get('WANDB_MODE', 'offline')

cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(cache_dir, exist_ok=True)
wandb_log_dir = os.path.join(cache_dir, 'wandb_runs')
os.makedirs(wandb_log_dir, exist_ok=True)
MODEL_TYPES = ['vanilla', 'relu', 'batchnorm', 'resnet']
NETRON_HOST = os.environ.get('NETRON_HOST', '127.0.0.1')
_netron_flag = os.environ.get('ENABLE_NETRON_ON_LOCAL_SERVER')
# Default: disable Netron inside Hugging Face Spaces (SPACE_ID is set there)
# because loopback ports are not exposed; can be re-enabled via
# ENABLE_NETRON_ON_LOCAL_SERVER=1
NETRON_ENABLED_ON_LOCAL_SERVER = (
    _netron_flag.lower()
    in ('1', 'true', 'yes')) if _netron_flag is not None else (
        not bool(os.environ.get('SPACE_ID')))
netron_servers = {}
netron_lock = threading.Lock()
REMOTE_MODEL_BASE_URL = os.environ.get(
    'REMOTE_MODEL_BASE_URL',
    'https://github.com/vinay-hebb/VanishingGradient/blob/main/cache/').strip()
# REMOTE_MODEL_BASE_URL = ''    # To render with on-disk models
# print(REMOTE_MODEL_BASE_URL)


def format_learning_rate(value):
    formatted = format(value, '.8f').rstrip('0').rstrip('.')
    return formatted if formatted else '0'


def get_cache_paths(key):
    path = os.path.join(cache_dir, key)
    return path, os.path.join(path, 'results.json')


def get_wandb_run_name(cache_key, model_type):
    return f'{cache_key}/{model_type}'


def get_wandb_run_dir(cache_key, model_type):
    return os.path.join(wandb_log_dir, cache_key, model_type)


def wandb_has_data():
    if not os.path.exists(wandb_log_dir):
        return False
    for _, _, files in os.walk(wandb_log_dir):
        if files:
            return True
    return False


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((NETRON_HOST, 0))
        return sock.getsockname()[1]


def get_remote_model_url(cache_key, filename):
    if not REMOTE_MODEL_BASE_URL:
        return None
    base = REMOTE_MODEL_BASE_URL
    if not base.endswith('/'):
        base += '/'
    # urljoin trims the filename if the base already ends with a filename, so
    # ensure we have a trailing slash above to treat it as a directory.
    return urljoin(base, f'{cache_key}/{filename}')


def start_netron_server(model_path=None):
    if not NETRON_AVAILABLE or not NETRON_ENABLED_ON_LOCAL_SERVER:
        if not NETRON_ENABLED_ON_LOCAL_SERVER:
            app.logger.info('Netron disabled; using netron.app embed instead.')
        return None

    abs_path = os.path.abspath(model_path) if model_path else None
    server_key = abs_path or '__netron_base__'
    with netron_lock:
        existing = netron_servers.get(server_key)
        if existing:
            return existing['url']

        try:
            port = find_available_port()
        except OSError as exc:
            app.logger.warning('Skipping Netron startup for %s: %s', abs_path,
                               exc)
            return None

        def run_netron():
            try:
                netron.start(abs_path,
                             address=(NETRON_HOST, port),
                             browse=False)
            except Exception as exc:
                app.logger.warning(
                    'Failed to start Netron for %s on port %s: %s',
                    abs_path, port, exc)

        thread = threading.Thread(target=run_netron, daemon=True)
        thread.start()
        netron_url = f'http://{NETRON_HOST}:{port}'
        app.logger.info('Netron server started for %s at %s',
                        abs_path or 'base server', netron_url)
        netron_servers[server_key] = {
            'url': netron_url,
            'port': port,
            'thread': thread
        }
        return netron_url


def get_wandb_instructions():
    if WANDB_MODE == 'offline':
        return f"Offline mode enabled. Run `wandb sync --sync-all {wandb_log_dir}` to upload logs when you're back online."
    base_url = os.environ.get('WANDB_BASE_URL', 'https://wandb.ai').rstrip('/')
    if base_url.endswith('/api'):
        base_url = base_url.rsplit('/', 1)[0]
    return (
        f"Runs stream live to {base_url}. Local copies are saved under {wandb_log_dir}; "
        "if connectivity drops you can still upload later with `wandb sync --sync-all <run_dir>`."
    )


ARTIFACT_CACHE_PATTERN = re.compile(
    r'^layers_(?P<num_layers>\d+)_epochs_(?P<num_epochs>\d+)_lr_(?P<learning_rate>[0-9.]+)_hidden_(?P<hidden_size>\d+)$'
)


def parse_cache_key(cache_key):
    match = ARTIFACT_CACHE_PATTERN.match(cache_key)
    if not match:
        return None
    try:
        params = match.groupdict()
        params['num_layers'] = int(params['num_layers'])
        params['num_epochs'] = int(params['num_epochs'])
        params['hidden_size'] = int(params['hidden_size'])
        params['learning_rate'] = float(params['learning_rate'])
        return params
    except (TypeError, ValueError):
        return None


def describe_artifact(params):
    if not params:
        return 'Unknown artifact'
    lr_display = format_learning_rate(params.get('learning_rate', 0.0))
    return (f"{params.get('num_layers', '?')} layers · "
            f"{params.get('hidden_size', '?')} hidden · "
            f"{params.get('num_epochs', '?')} epochs · "
            f"lr {lr_display}")


def build_artifact_metadata(cache_key, cached_data=None):
    params = parse_cache_key(cache_key)
    models = sorted(cached_data.keys()) if cached_data else []
    return {
        'id': cache_key,
        'params': params,
        'label': describe_artifact(params),
        'models': models
    }


def load_cached_results(cache_key):
    cache_path, cache_results_file = get_cache_paths(cache_key)
    if not os.path.exists(cache_results_file):
        return None

    try:
        with open(cache_results_file, 'r') as cache_file:
            cached_data = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return None

    wandb_payload = {
        'enabled': bool(WANDB_AVAILABLE),
        'mode': WANDB_MODE,
        'storage_dir': wandb_log_dir,
        'runs': {},
        'has_data': False,
        'instructions': get_wandb_instructions()
    }
    results = {}

    netron_url = start_netron_server()

    for model_type in MODEL_TYPES:
        cached_entry = cached_data.get(model_type)
        if not cached_entry:
            continue
        cached_model_filename = cached_entry.get('model_file')
        cached_model_path = os.path.join(
            cache_path, cached_model_filename) if cached_model_filename else None
        model_exists = bool(cached_model_path
                            and os.path.exists(cached_model_path))

        entry = cached_entry.copy()
        entry['model_file'] = cached_model_filename
        entry['model_available'] = model_exists
        if cached_model_filename:
            remote_url = get_remote_model_url(cache_key, cached_model_filename)
            if remote_url:
                entry['remote_model_url'] = remote_url
                app.logger.info('Netron remote URL for %s/%s set to %s',
                                cache_key, model_type, remote_url)
        if model_exists:
            entry['model_url'] = (
                f'/api/artifacts/{cache_key}/model/{cached_model_filename}')
            if netron_url:
                entry['netron_url'] = netron_url
        else:
            entry.pop('model_url', None)
            entry.pop('netron_url', None)
        wandb_run_name = entry.get('wandb_run_name') or get_wandb_run_name(
            cache_key, model_type)
        entry['wandb_run_name'] = wandb_run_name
        entry['wandb_run'] = entry.get('wandb_run') or get_wandb_run_dir(
            cache_key, model_type)
        wandb_payload['runs'][model_type] = {
            'name': wandb_run_name,
            'path': entry['wandb_run'],
            'url': entry.get('wandb_url')
        }
        results[model_type] = entry

    if not results:
        return None

    wandb_payload['has_data'] = bool(wandb_payload['enabled']
                                     and wandb_has_data())
    artifact_meta = build_artifact_metadata(cache_key, results)
    return {
        'results': results,
        'wandb': wandb_payload,
        'artifact': artifact_meta
    }


def list_available_artifacts():
    artifacts = []
    if not os.path.exists(cache_dir):
        return artifacts

    for entry in sorted(os.listdir(cache_dir)):
        cache_key = os.path.basename(entry)
        params = parse_cache_key(cache_key)
        if not params:
            continue
        _, cache_results_file = get_cache_paths(cache_key)
        if not os.path.exists(cache_results_file):
            continue
        try:
            with open(cache_results_file, 'r') as cache_file:
                cached_data = json.load(cache_file)
        except (OSError, json.JSONDecodeError):
            continue
        artifacts.append(build_artifact_metadata(cache_key, cached_data))
    return artifacts


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/artifacts', methods=['GET'])
def list_artifacts():
    artifacts = list_available_artifacts()
    return jsonify({'artifacts': artifacts})


@app.route('/api/artifacts/<path:cache_key>', methods=['GET'])
def fetch_artifact(cache_key):
    cache_key = os.path.basename(cache_key)
    if not parse_cache_key(cache_key):
        return jsonify({'error': 'Artifact not found'}), 404

    payload = load_cached_results(cache_key)
    if not payload:
        return jsonify({'error': 'Artifact not found'}), 404

    return jsonify(payload)


@app.route('/api/artifacts/<path:cache_key>/model/<path:filename>')
def get_model_file(cache_key, filename):
    cache_key = os.path.basename(cache_key)
    filename = os.path.basename(filename)

    if not parse_cache_key(cache_key):
        return jsonify({'error': 'Artifact not found'}), 404

    if not filename.endswith('.onnx'):
        return jsonify({'error': 'Invalid file type'}), 400

    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'Invalid filename'}), 400

    cache_path, cache_results_file = get_cache_paths(cache_key)
    if not os.path.exists(cache_results_file):
        return jsonify({'error': 'Artifact not found'}), 404

    filepath = os.path.join(cache_path, filename)

    if os.path.exists(filepath) and os.path.isfile(filepath):
        return send_file(filepath, mimetype='application/octet-stream')
    return jsonify({'error': 'Model file not found'}), 404


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    debug_env = os.environ.get('FLASK_DEBUG', '')
    debug_mode = debug_env.lower() in ('1', 'true', 'yes')
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=debug_mode)
