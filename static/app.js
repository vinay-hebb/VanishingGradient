let trainingData = null;
let currentModelType = null;
let currentHeatmapModel = 'vanilla';
let animationInterval = null;
let currentAnimationEpoch = 0;
let gradientsChart = null;
let lossChart = null;
let gradientScaleType = 'logarithmic';
let wandbInfo = null;
let availableArtifacts = [];
let currentArtifact = null;

const modelColors = {
    vanilla: '#e74c3c',
    relu: '#3498db',
    batchnorm: '#2ecc71',
    resnet: '#f39c12'
};

const modelNames = {
    vanilla: 'Vanilla (Sigmoid)',
    relu: 'ReLU Activation',
    batchnorm: 'BatchNorm + ReLU',
    resnet: 'ResNet (Skip Connections)'
};

document.addEventListener('DOMContentLoaded', () => {
    const animateBtn = document.getElementById('animate-btn');
    if (animateBtn) {
        animateBtn.addEventListener('click', toggleAnimation);
    }

    const artifactSelect = document.getElementById('artifact-set');
    if (artifactSelect) {
        artifactSelect.addEventListener('change', (event) => {
            const cacheKey = event.target.value;
            if (cacheKey) {
                loadArtifact(cacheKey);
            }
        });
    }

    fetchArtifacts();
});

function setLoading(isLoading, message = '') {
    const loadingEl = document.getElementById('loading');
    if (!loadingEl) return;
    loadingEl.classList.toggle('active', isLoading);
    const text = loadingEl.querySelector('p');
    if (text && message) {
        text.textContent = message;
    }
}

function updateStatus(message, tone = 'muted') {
    const statusEl = document.getElementById('artifact-status');
    if (!statusEl) return;
    statusEl.textContent = message || '';
    statusEl.dataset.tone = tone;
}

async function fetchArtifacts() {
    setLoading(true, 'Scanning for artifacts...');
    updateStatus('');
    try {
        const response = await fetch('/api/artifacts');
        if (!response.ok) {
            throw new Error(`${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        availableArtifacts = data.artifacts || [];
        renderArtifactOptions();
        if (availableArtifacts.length) {
            const first = availableArtifacts[0];
            loadArtifact(first.id);
        } else {
            handleNoArtifacts();
        }
    } catch (error) {
        console.error('Unable to load artifacts', error);
        updateStatus('Unable to list artifacts. Verify the cache directory.', 'error');
        handleNoArtifacts();
    } finally {
        setLoading(false);
    }
}

function renderArtifactOptions() {
    const select = document.getElementById('artifact-set');
    if (!select) return;
    select.innerHTML = '';

    if (!availableArtifacts.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No artifacts detected';
        select.appendChild(option);
        select.disabled = true;
        return;
    }

    availableArtifacts.forEach((artifact, index) => {
        const option = document.createElement('option');
        option.value = artifact.id;
        option.textContent = artifact.label || artifact.id;
        if (index === 0) {
            option.selected = true;
        }
        select.appendChild(option);
    });
    select.disabled = false;
}

function handleNoArtifacts() {
    stopAnimation();
    const animateBtn = document.getElementById('animate-btn');
    if (animateBtn) {
        animateBtn.disabled = true;
    }
    const results = document.getElementById('results');
    if (results) {
        results.style.display = 'none';
    }
    updateStatus('No artifact sets found. Run training separately to populate the cache.', 'warning');
}

async function loadArtifact(cacheKey) {
    setLoading(true, 'Loading artifact...');
    stopAnimation();
    updateStatus('');
    const animateBtn = document.getElementById('animate-btn');
    if (animateBtn) {
        animateBtn.disabled = true;
    }

    try {
        const response = await fetch(`/api/artifacts/${encodeURIComponent(cacheKey)}`);
        if (!response.ok) {
            throw new Error(`Artifact ${cacheKey} not found`);
        }
        const data = await response.json();
        trainingData = data.results;
        wandbInfo = data.wandb || null;
        currentArtifact = data.artifact || null;
        currentModelType = null;
        currentHeatmapModel = 'vanilla';

        renderArtifactMeta();
        setupModelTabs();
        setupHeatmapTabs();
        renderGradientsChart();
        renderLossChart();
        renderHeatmap();
        renderWandbPanel();
        showNetronPlaceholder();
        document.getElementById('results').style.display = 'block';
        updateStatus('Artifact loaded. Explore the visualizations.', 'success');
    } catch (error) {
        console.error('Failed to load artifact', error);
        updateStatus(error.message, 'error');
    } finally {
        setLoading(false);
    }
}

function renderArtifactMeta() {
    const metaEl = document.getElementById('artifact-meta');
    if (!metaEl) return;
    if (!currentArtifact || !currentArtifact.params) {
        metaEl.textContent = '';
        return;
    }
    const params = currentArtifact.params;
    const learningRate = params.learning_rate;
    metaEl.textContent = [
        `${params.num_layers} layers`,
        `${params.hidden_size} hidden`,
        `${params.num_epochs} epochs`,
        `lr ${learningRate}`
    ].join(' • ');
}

function setupModelTabs() {
    const tabsContainer = document.getElementById('model-tabs');
    if (!tabsContainer || !trainingData) return;
    tabsContainer.innerHTML = '';

    Object.keys(modelNames).forEach(modelType => {
        const tab = document.createElement('button');
        tab.type = 'button';
        tab.className = 'model-tab' + (modelType === currentModelType ? ' active' : '');
        tab.textContent = modelNames[modelType];
        tab.addEventListener('click', () => {
            currentModelType = modelType;
            updateModelTabs();
            loadModelVisualization(modelType);
        });
        tabsContainer.appendChild(tab);
    });
}

function setupHeatmapTabs() {
    const tabsContainer = document.getElementById('heatmap-tabs');
    if (!tabsContainer || !trainingData) return;
    tabsContainer.innerHTML = '';

    Object.keys(modelNames).forEach(modelType => {
        const tab = document.createElement('button');
        tab.type = 'button';
        tab.className = 'model-tab' + (modelType === currentHeatmapModel ? ' active' : '');
        tab.textContent = modelNames[modelType];
        tab.addEventListener('click', () => {
            currentHeatmapModel = modelType;
            updateHeatmapTabs();
            renderHeatmap();
        });
        tabsContainer.appendChild(tab);
    });
}

function updateModelTabs() {
    const tabs = document.querySelectorAll('#model-tabs .model-tab');
    tabs.forEach((tab, index) => {
        const modelType = Object.keys(modelNames)[index];
        tab.classList.toggle('active', modelType === currentModelType);
    });
}

function updateHeatmapTabs() {
    const tabs = document.querySelectorAll('#heatmap-tabs .model-tab');
    tabs.forEach((tab, index) => {
        const modelType = Object.keys(modelNames)[index];
        tab.classList.toggle('active', modelType === currentHeatmapModel);
    });
}

function showNetronPlaceholder() {
    const netronView = document.getElementById('netron-view');
    if (!netronView) return;
    netronView.innerHTML =
        '<p class="muted">Select a model tab to load the architecture view.</p>';
}

function loadModelVisualization(modelType) {
    if (!trainingData || !modelType) {
        showNetronPlaceholder();
        return;
    }
    const netronView = document.getElementById('netron-view');
    const modelData = trainingData[modelType];
    const remoteModelUrl = modelData?.remote_model_url;
    const modelUrl = remoteModelUrl || modelData?.model_url;
    const netronUrl = modelData?.netron_url;
    if (!netronView) return;

    if (!modelUrl) {
        netronView.innerHTML = `<p class="muted">Model file not available for ${modelNames[modelType]}.</p>`;
        return;
    }

    netronView.innerHTML = '<p class="muted">Loading model architecture...</p>';

    const fullModelUrl = new URL(modelUrl, window.location.origin).toString();
    let iframeUrl;
    if (netronUrl) {
        const baseNetronUrl = netronUrl.replace(/\/$/, '');
        iframeUrl = `${baseNetronUrl}/?url=${encodeURIComponent(fullModelUrl)}`;
        console.log(`Loading Netron from local server: ${iframeUrl}, ${modelType}`);
    } else {
        console.log(`Loading model visualization via netron.app for: ${fullModelUrl}, ${modelType}`);
        iframeUrl = `https://netron.app/?url=${encodeURIComponent(fullModelUrl)}`;
    }

    const iframe = document.createElement('iframe');
    iframe.src = iframeUrl;
    iframe.className = 'netron-frame';
    iframe.title = `${modelNames[modelType]} architecture`;
    iframe.loading = 'lazy';

    iframe.addEventListener('error', () => {
        netronView.innerHTML = `<p class="muted">Unable to load Netron viewer for ${modelNames[modelType]}.</p>`;
    });

    netronView.innerHTML = '';
    netronView.appendChild(iframe);
}

function renderGradientsChart(epochIndex = null) {
    if (!trainingData) return;
    const ctx = document.getElementById('gradients-chart').getContext('2d');

    if (gradientsChart) {
        gradientsChart.destroy();
    }

    const datasets = [];
    const isAnimating = epochIndex !== null;

    Object.keys(modelNames).forEach(modelType => {
        const data = trainingData[modelType];
        let gradientData;

        if (isAnimating && data.heatmap_data && data.heatmap_data.data) {
            gradientData = data.heatmap_data.data.map(layerGrads => layerGrads[epochIndex]);
        } else {
            gradientData = data.final_gradients;
        }

        if (gradientData) {
            datasets.push({
                label: modelNames[modelType],
                data: gradientData,
                borderColor: modelColors[modelType],
                backgroundColor: modelColors[modelType] + '20',
                borderWidth: 3,
                pointRadius: 4,
                pointHoverRadius: 7,
                tension: 0.2
            });
        }
    });

    const firstModel = Object.keys(trainingData)[0];
    const labels = trainingData[firstModel].layer_names?.map((name, idx) => `L${idx + 1}`) || [];

    gradientsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Gradient Magnitude by Layer'
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                annotation: {
                    annotations: {}
                }
            },
            scales: {
                y: {
                    type: gradientScaleType,
                    title: {
                        display: true,
                        text: 'Gradient Magnitude'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Layer'
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            onHover: (event, elements) => {
                if (elements.length > 0) {
                    const layerIndex = elements[0].index;
                    highlightLayer(layerIndex);
                } else {
                    clearLayerHighlight();
                }
            }
        }
    });
}

function renderLossChart() {
    if (!trainingData) return;
    const ctx = document.getElementById('loss-chart').getContext('2d');

    if (lossChart) {
        lossChart.destroy();
    }

    const datasets = [];

    Object.keys(modelNames).forEach(modelType => {
        const data = trainingData[modelType];
        if (data.loss_history) {
            datasets.push({
                label: modelNames[modelType],
                data: data.loss_history,
                borderColor: modelColors[modelType],
                backgroundColor: modelColors[modelType] + '20',
                borderWidth: 3,
                pointRadius: 4,
                pointHoverRadius: 7,
                tension: 0.3
            });
        }
    });

    const epochs = trainingData[Object.keys(trainingData)[0]].loss_history.length;
    const labels = Array.from({ length: epochs }, (_, i) => i + 1);

    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Training Loss Over Epochs'
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            onHover: (event, elements) => {
                if (elements.length > 0) {
                    const epochIndex = elements[0].index;
                    showLossInfo(epochIndex);
                } else {
                    clearLossInfo();
                }
            }
        }
    });
}

function renderHeatmap() {
    if (!trainingData) return;
    const canvas = document.getElementById('heatmap-canvas');
    const ctx = canvas.getContext('2d');
    const heatmapData = trainingData[currentHeatmapModel].heatmap_data;

    if (!heatmapData || !heatmapData.data) {
        canvas.width = 800;
        canvas.height = 400;
        ctx.fillStyle = '#333';
        ctx.font = '16px Arial';
        ctx.fillText('No heatmap data available', 20, 200);
        return;
    }

    const data = heatmapData.data;
    const epochs = heatmapData.epochs;
    const layerNames = heatmapData.layer_names;

    const numLayers = data.length;
    const numEpochs = epochs.length;

    const cellWidth = 40;
    const cellHeight = 30;
    const marginLeft = 100;
    const marginTop = 40;
    const marginBottom = 50;

    canvas.width = marginLeft + numEpochs * cellWidth + 80;
    canvas.height = marginTop + numLayers * cellHeight + marginBottom;

    let maxGrad = 0;
    data.forEach(layerGrads => {
        layerGrads.forEach(grad => {
            maxGrad = Math.max(maxGrad, grad);
        });
    });

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        for (let epochIdx = 0; epochIdx < numEpochs; epochIdx++) {
            const grad = data[layerIdx][epochIdx];
            const normalizedGrad = maxGrad === 0 ? 0 : grad / maxGrad;

            const hue = 240 - (normalizedGrad * 240);
            const color = `hsl(${hue}, 70%, 50%)`;

            ctx.fillStyle = color;
            ctx.fillRect(
                marginLeft + epochIdx * cellWidth,
                marginTop + layerIdx * cellHeight,
                cellWidth - 1,
                cellHeight - 1
            );
        }
    }

    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < numLayers; i++) {
        const layerName = layerNames?.[i] || `L${i + 1}`;
        ctx.fillText(layerName, marginLeft - 10, marginTop + i * cellHeight + cellHeight / 2);
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i < numEpochs; i += Math.max(1, Math.ceil(numEpochs / 8))) {
        ctx.fillText(
            epochs[i].toString(),
            marginLeft + i * cellWidth + cellWidth / 2,
            marginTop + numLayers * cellHeight + 10
        );
    }

    ctx.textAlign = 'center';
    ctx.font = 'bold 13px Arial';
    ctx.fillText('Epochs', marginLeft + numEpochs * cellWidth / 2, canvas.height - 12);

    ctx.save();
    ctx.translate(20, marginTop + numLayers * cellHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Layers', 0, 0);
    ctx.restore();

    canvas.onmousemove = (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const layerIdx = Math.floor((y - marginTop) / cellHeight);
        const epochIdx = Math.floor((x - marginLeft) / cellWidth);

        if (layerIdx >= 0 && layerIdx < numLayers && epochIdx >= 0 && epochIdx < numEpochs) {
            showHeatmapInfo(layerIdx, epochIdx);
            highlightLayer(layerIdx);
        } else {
            clearHeatmapInfo();
            clearLayerHighlight();
        }
    };

    canvas.onmouseleave = () => {
        clearHeatmapInfo();
        clearLayerHighlight();
    };
}

function renderWandbPanel() {
    const buttonsElement = document.getElementById('wandb-buttons');
    const emptyStateElement = document.getElementById('wandb-empty');
    if (!buttonsElement || !emptyStateElement) return;

    buttonsElement.innerHTML = '';
    emptyStateElement.textContent = 'Load an artifact set to see W&B runs.';
    emptyStateElement.style.display = 'block';

    if (!wandbInfo || !wandbInfo.enabled) {
        return;
    }

    const runEntries = Object.entries(wandbInfo.runs || {});
    if (!runEntries.length) {
        emptyStateElement.textContent = 'Weights & Biases links will appear after logging.';
        return;
    }

    let hasNavigableLink = false;
    runEntries.forEach(([modelKey, info]) => {
        const label = modelNames[modelKey] || modelKey;
        const runButton = document.createElement('button');
        runButton.type = 'button';
        runButton.className = 'btn';
        runButton.textContent = label;

        const runUrl = info?.url;
        if (runUrl) {
            hasNavigableLink = true;
            runButton.addEventListener('click', () => {
                window.open(runUrl, '_blank', 'noopener');
            });
        } else {
            runButton.disabled = true;
            runButton.title = 'Run URL not available yet.';
        }

        buttonsElement.appendChild(runButton);
    });

    if (!hasNavigableLink) {
        emptyStateElement.textContent = 'Links will appear once wandb provides run URLs.';
    } else {
        emptyStateElement.style.display = 'none';
    }
}

function highlightLayer(layerIndex) {
    if (!trainingData) return;
    const layerInfo = document.getElementById('layer-info');
    const firstModel = Object.keys(trainingData)[0];
    const layerName = trainingData[firstModel].layer_names?.[layerIndex] || `Layer ${layerIndex + 1}`;

    layerInfo.innerHTML = `
        <strong>Layer ${layerIndex + 1}</strong>: ${layerName}<br>
        Gradient magnitudes:
        ${Object.keys(modelNames).map(modelType => {
            const grad = trainingData[modelType].final_gradients?.[layerIndex];
            const value = grad ? grad.toExponential(3) : 'N/A';
            return `<br><span style="color: ${modelColors[modelType]}">● ${modelNames[modelType]}: ${value}</span>`;
        }).join('')}
    `;
    layerInfo.classList.add('active');

    if (gradientsChart) {
        const annotation = {
            type: 'line',
            mode: 'vertical',
            scaleID: 'x',
            value: layerIndex,
            borderColor: '#ff6384',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                content: `Layer ${layerIndex + 1}`,
                enabled: true,
                position: 'top'
            }
        };

        if (!gradientsChart.options.plugins.annotation) {
            gradientsChart.options.plugins.annotation = { annotations: {} };
        }
        gradientsChart.options.plugins.annotation.annotations = {
            line1: annotation
        };
        gradientsChart.update('none');
    }
}

function clearLayerHighlight() {
    const layerInfo = document.getElementById('layer-info');
    layerInfo.classList.remove('active');

    if (gradientsChart && gradientsChart.options.plugins.annotation) {
        gradientsChart.options.plugins.annotation.annotations = {};
        gradientsChart.update('none');
    }
}

function toggleAnimation() {
    if (animationInterval) {
        stopAnimation();
    } else {
        startAnimation();
    }
}

function startAnimation() {
    if (!trainingData) return;
    const heatmapData = trainingData[currentHeatmapModel].heatmap_data;
    if (!heatmapData || !heatmapData.epochs) return;

    const numEpochs = heatmapData.epochs.length;
    currentAnimationEpoch = 0;

    const animateBtn = document.getElementById('animate-btn');
    if (animateBtn) {
        animateBtn.textContent = 'Pause';
    }

    animationInterval = setInterval(() => {
        updateAnimationFrame();
        currentAnimationEpoch++;

        if (currentAnimationEpoch >= numEpochs) {
            currentAnimationEpoch = 0;
        }
    }, 200);
}

function stopAnimation() {
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
        const animateBtn = document.getElementById('animate-btn');
        if (animateBtn) {
            animateBtn.textContent = 'Animate';
        }
    }
}

function updateAnimationFrame() {
    if (!trainingData) return;
    const heatmapData = trainingData[currentHeatmapModel].heatmap_data;
    if (!heatmapData) return;

    const epochs = heatmapData.epochs;
    const data = heatmapData.data;
    const numEpochs = epochs.length;

    const progress = (currentAnimationEpoch / (numEpochs - 1)) * 100;
    document.getElementById('progress-fill').style.width = `${progress}%`;
    document.getElementById('epoch-display').textContent = `${epochs[currentAnimationEpoch]} / ${epochs[numEpochs - 1]}`;

    const currentGradients = data.map(layerGrads => layerGrads[currentAnimationEpoch]);

    if (gradientsChart) {
        const datasetIndex = Object.keys(modelNames).indexOf(currentHeatmapModel);
        if (datasetIndex >= 0 && gradientsChart.data.datasets[datasetIndex]) {
            gradientsChart.data.datasets.forEach((dataset, idx) => {
                if (idx === datasetIndex) {
                    dataset.data = currentGradients;
                    dataset.borderWidth = 5;
                } else {
                    dataset.borderWidth = 1;
                }
            });
            gradientsChart.update('none');
        }
    }
}

function showLossInfo(epochIndex) {
    if (!trainingData) return;
    const lossInfo = document.getElementById('loss-info');

    lossInfo.innerHTML = `
        <strong>Epoch ${epochIndex + 1}</strong><br>
        Loss values:
        ${Object.keys(modelNames).map(modelType => {
            const loss = trainingData[modelType].loss_history?.[epochIndex];
            const value = typeof loss === 'number' ? loss.toFixed(4) : 'N/A';
            return `<br><span style="color: ${modelColors[modelType]}">● ${modelNames[modelType]}: ${value}</span>`;
        }).join('')}
    `;
    lossInfo.classList.add('active');
}

function clearLossInfo() {
    const lossInfo = document.getElementById('loss-info');
    lossInfo.classList.remove('active');
}

function showHeatmapInfo(layerIdx, epochIdx) {
    if (!trainingData) return;
    const heatmapInfo = document.getElementById('heatmap-layer-info');
    const data = trainingData[currentHeatmapModel].heatmap_data;

    if (!data || !data.data) return;

    const grad = data.data[layerIdx][epochIdx];
    const epoch = data.epochs[epochIdx];
    const layerName = data.layer_names?.[layerIdx] || `Layer ${layerIdx + 1}`;

    heatmapInfo.innerHTML = `
        <strong>Layer ${layerIdx + 1}</strong> (${layerName}), <strong>Epoch ${epoch}</strong><br>
        <span style="color: ${modelColors[currentHeatmapModel]}">● ${modelNames[currentHeatmapModel]}: Gradient = ${grad.toExponential(3)}</span>
    `;
    heatmapInfo.classList.add('active');
}

function clearHeatmapInfo() {
    const heatmapInfo = document.getElementById('heatmap-layer-info');
    heatmapInfo.classList.remove('active');
}
