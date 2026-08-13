const form = document.getElementById('prediction-form');
const input = document.getElementById('image-input');
const dropZone = document.getElementById('drop-zone');
const previewWrap = document.getElementById('preview-wrap');
const preview = document.getElementById('image-preview');
const removeImage = document.getElementById('remove-image');
const predictButton = document.getElementById('predict-button');
const formMessage = document.getElementById('form-message');
const serviceStatus = document.getElementById('service-status');
const resultPlaceholder = document.getElementById('result-placeholder');
const resultContent = document.getElementById('result-content');

let selectedFile = null;

function setMessage(message = '', isError = false) {
  formMessage.textContent = message;
  formMessage.classList.toggle('error', isError);
}

function resetResult() {
  resultContent.hidden = true;
  resultPlaceholder.hidden = false;
}

function clearImage() {
  selectedFile = null;
  input.value = '';
  preview.src = '';
  previewWrap.hidden = true;
  dropZone.hidden = false;
  predictButton.disabled = true;
  setMessage('');
  resetResult();
}

function selectFile(file) {
  if (!file) return;
  const allowed = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
  if (!allowed.includes(file.type)) {
    clearImage();
    setMessage('Please select a supported image file.', true);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    clearImage();
    setMessage('The selected image is larger than 10 MB.', true);
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  previewWrap.hidden = false;
  dropZone.hidden = true;
  predictButton.disabled = false;
  setMessage(`${file.name} is ready for analysis.`);
  resetResult();
}

function listItems(container, values) {
  container.innerHTML = '';
  values.forEach((value) => {
    const item = document.createElement('li');
    item.textContent = value;
    container.appendChild(item);
  });
}

function renderPrediction(data) {
  document.getElementById('disease-name').textContent = data.disease || 'Unknown';
  document.getElementById('confidence-value').textContent = data.confidence || '—';
  document.getElementById('severity-value').textContent = data.severity || '—';
  document.getElementById('action-value').textContent = data.action || 'No action information available.';

  const chemicalsBlock = document.getElementById('chemical-block');
  const preventionBlock = document.getElementById('prevention-block');
  chemicalsBlock.hidden = !(data.chemicals && data.chemicals.length);
  preventionBlock.hidden = !(data.prevention && data.prevention.length);
  if (!chemicalsBlock.hidden) listItems(document.getElementById('chemicals-list'), data.chemicals);
  if (!preventionBlock.hidden) listItems(document.getElementById('prevention-list'), data.prevention);

  const probabilities = document.getElementById('probability-list');
  probabilities.innerHTML = '';
  Object.entries(data.all_predictions || {}).forEach(([name, score]) => {
    const raw = Number.parseFloat(score);
    const row = document.createElement('div');
    row.className = 'probability-row';
    row.innerHTML = `<span>${name}</span><span>${score}</span><div class="meter"><span style="width:${Math.max(0, Math.min(100, raw || 0))}%"></span></div>`;
    probabilities.appendChild(row);
  });

  resultPlaceholder.hidden = true;
  resultContent.hidden = false;
}

async function checkHealth() {
  try {
    const response = await fetch('/health', { cache: 'no-store' });
    const data = await response.json();
    if (data.model_loaded) {
      serviceStatus.textContent = 'Service ready';
      serviceStatus.className = 'status-pill ready';
    } else {
      serviceStatus.textContent = 'Model unavailable';
      serviceStatus.className = 'status-pill error';
      predictButton.disabled = true;
    }
  } catch (error) {
    serviceStatus.textContent = 'Service unavailable';
    serviceStatus.className = 'status-pill error';
  }
}

input.addEventListener('change', () => selectFile(input.files[0]));
removeImage.addEventListener('click', clearImage);

['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add('drag-active');
  });
});
['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove('drag-active');
  });
});
dropZone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!selectedFile) return;

  predictButton.disabled = true;
  predictButton.textContent = 'Analysing image…';
  setMessage('');

  const payload = new FormData();
  payload.append('file', selectedFile);

  try {
    const response = await fetch('/predict', { method: 'POST', body: payload });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The image could not be analysed.');
    renderPrediction(data);
    setMessage('Analysis complete. Review the result and recommendations.');
  } catch (error) {
    resetResult();
    setMessage(error.message || 'The image could not be analysed.', true);
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = 'Analyse image';
  }
});

checkHealth();
