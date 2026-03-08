// ===================== Theme Toggle =====================
// Apply saved theme immediately to prevent flash
(function () {
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-mode');
    }
})();

function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

let selectedSymptoms = [];
let customSymptoms = [];
let allSymptoms = [];
let selectedFile = null;


// Initialize app
document.addEventListener('DOMContentLoaded', function () {
    // Only load symptoms if we're on a page with the symptom grid
    if (document.getElementById('symptoms-grid')) {
        loadSymptoms();
    }
    setupEventListeners();
});

// Smooth scroll to section
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Load all symptoms from backend
async function loadSymptoms() {
    try {
        const response = await fetch('/get-symptoms');
        const data = await response.json();
        allSymptoms = data.symptoms;
        displaySymptomGrid();
    } catch (error) {
        console.error('Error loading symptoms:', error);
        showError('Failed to load symptoms. Please refresh the page.');
    }
}

// Display symptom checkboxes
function displaySymptomGrid() {
    const grid = document.getElementById('symptoms-grid');
    if (!grid) return; // Exit if grid doesn't exist

    grid.innerHTML = allSymptoms.map(symptom => `
        <div class="symptom-checkbox" id="symptom-${symptom.replace(/\s+/g, '-')}">
            <input type="checkbox" id="check-${symptom.replace(/\s+/g, '-')}" onchange="toggleSymptom('${symptom}')">
            <label for="check-${symptom.replace(/\s+/g, '-')}">${symptom}</label>
        </div>
    `).join('');
}

// Add custom symptom
function addCustomSymptom() {
    const input = document.getElementById('custom-symptom-input');
    const symptomText = input.value.trim();

    if (!symptomText) {
        showError('Please enter a symptom');
        return;
    }

    // Convert to title case for consistency
    const symptomFormatted = symptomText.toLowerCase().split(' ').map(word =>
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');

    // Check if already added
    if (customSymptoms.includes(symptomFormatted) || selectedSymptoms.includes(symptomFormatted)) {
        showError('This symptom is already added');
        return;
    }

    // Add to custom symptoms
    customSymptoms.push(symptomFormatted);
    updateSelectedSymptoms();

    // Clear input
    input.value = '';
}

// Setup event listeners
function setupEventListeners() {
    // Symptom search
    const symptomSearch = document.getElementById('symptom-search');
    if (symptomSearch) {
        symptomSearch.addEventListener('input', handleSymptomSearch);
    }

    // Custom symptom input - allow Enter key
    const customInput = document.getElementById('custom-symptom-input');
    if (customInput) {
        customInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addCustomSymptom();
            }
        });
    }

    // Image upload
    const uploadArea = document.getElementById('upload-area');
    const imageInput = document.getElementById('image-input');

    if (uploadArea && imageInput) {
        uploadArea.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', handleImageSelect);

        // Drag and drop
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }

    // Navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href && href.startsWith('#')) {
                e.preventDefault();
                const sectionId = href.substring(1);
                scrollToSection(sectionId);

                // Update active state
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });
}

// Handle symptom search
function handleSymptomSearch(e) {
    const searchTerm = e.target.value.toLowerCase();
    const checkboxes = document.querySelectorAll('.symptom-checkbox');

    checkboxes.forEach(checkbox => {
        const label = checkbox.querySelector('label');
        const symptomName = label.textContent.toLowerCase();

        if (symptomName.includes(searchTerm)) {
            checkbox.style.display = 'flex';
        } else {
            checkbox.style.display = 'none';
        }
    });
}

// Toggle symptom selection
function toggleSymptom(symptom) {
    const checkbox = document.getElementById(`check-${symptom.replace(/\s+/g, '-')}`);
    const symptomDiv = document.getElementById(`symptom-${symptom.replace(/\s+/g, '-')}`);

    if (checkbox.checked) {
        if (!selectedSymptoms.includes(symptom)) {
            selectedSymptoms.push(symptom);
            symptomDiv.classList.add('selected');
        }
    } else {
        selectedSymptoms = selectedSymptoms.filter(s => s !== symptom);
        symptomDiv.classList.remove('selected');
    }

    updateSelectedSymptoms();
}

// Update selected symptoms display
function updateSelectedSymptoms() {
    const container = document.getElementById('selected-symptoms');
    const analyzeBtn = document.getElementById('analyze-symptoms-btn');
    const clearAllBtn = document.getElementById('clear-all-btn');

    const totalSymptoms = selectedSymptoms.length + customSymptoms.length;

    if (totalSymptoms === 0) {
        container.innerHTML = '<p class="no-symptoms">No symptoms selected yet</p>';
        if (analyzeBtn) analyzeBtn.disabled = true;
        if (clearAllBtn) clearAllBtn.style.display = 'none';
    } else {
        let html = '';

        // Display predefined symptoms
        html += selectedSymptoms
            .map(symptom => `
                <span class="symptom-tag">
                    ${symptom}
                    <button onclick="removeSymptom('${symptom}')">×</button>
                </span>
            `)
            .join('');

        // Display custom symptoms with different style
        html += customSymptoms
            .map(symptom => `
                <span class="symptom-tag custom-symptom-tag">
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="margin-right: 4px;">
                        <path d="M7 1v12M1 7h12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                    ${symptom}
                    <button onclick="removeCustomSymptom('${symptom}')">×</button>
                </span>
            `)
            .join('');

        container.innerHTML = html;
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (clearAllBtn) clearAllBtn.style.display = 'inline-flex';
    }
}

// Remove symptom
function removeSymptom(symptom) {
    const checkbox = document.getElementById(`check-${symptom.replace(/\s+/g, '-')}`);
    const symptomDiv = document.getElementById(`symptom-${symptom.replace(/\s+/g, '-')}`);

    if (checkbox) checkbox.checked = false;
    if (symptomDiv) symptomDiv.classList.remove('selected');

    selectedSymptoms = selectedSymptoms.filter(s => s !== symptom);
    updateSelectedSymptoms();
}

// Remove custom symptom
function removeCustomSymptom(symptom) {
    customSymptoms = customSymptoms.filter(s => s !== symptom);
    updateSelectedSymptoms();
}

// Clear all selected symptoms
function clearAllSymptoms() {
    selectedSymptoms = [];
    customSymptoms = [];
    // Uncheck all checkboxes in the grid
    document.querySelectorAll('.symptom-checkbox input').forEach(cb => cb.checked = false);
    document.querySelectorAll('.symptom-checkbox').forEach(div => div.classList.remove('selected'));
    updateSelectedSymptoms();
}

// Analyze symptoms
async function analyzeSymptoms() {
    const totalSymptoms = selectedSymptoms.length + customSymptoms.length;

    if (totalSymptoms === 0) {
        showError('Please select or enter at least one symptom');
        return;
    }

    const btn = document.getElementById('analyze-symptoms-btn');
    setBtnLoading(btn, true, 'Analyzing...');
    showLoading(true);

    try {
        // Combine both predefined and custom symptoms
        const allSelectedSymptoms = [...selectedSymptoms, ...customSymptoms];

        // Convert symptoms back to underscore format for API
        const symptomsForAPI = allSelectedSymptoms.map(s =>
            s.toLowerCase().replace(/ /g, '_')
        );

        const response = await fetch('/predict-symptoms', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symptoms: symptomsForAPI })
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred. Please try again.');
    } finally {
        setBtnLoading(btn, false, 'Analyze Symptoms');
        showLoading(false);
    }
}

// Handle image selection
function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processImageFile(file);
    }
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processImageFile(file);
    } else {
        showError('Please upload a valid image file');
    }
}

// Process image file
function processImageFile(file) {
    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = function (e) {
        const previewDiv = document.getElementById('image-preview');
        previewDiv.innerHTML = `
            <img src="${e.target.result}" alt="Preview">
            <p class="preview-name">📄 ${file.name}</p>
        `;
        previewDiv.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Enable submit button
    const analyzeBtn = document.getElementById('analyze-image-btn');
    if (analyzeBtn) {
        analyzeBtn.style.display = 'block';
        analyzeBtn.disabled = false;
    }
}



// Analyze image
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    const btn = document.getElementById('analyze-image-btn');
    setBtnLoading(btn, true, 'Analyzing...');
    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch('/predict-image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Image analysis failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred. Please try again.');
    } finally {
        setBtnLoading(btn, false, 'Analyze Image');
        showLoading(false);
    }
}

// Display results
function displayResults(data) {
    const resultsBody = document.getElementById('results-body');

    // Build warning section if there are unknown symptoms
    let warningSection = '';
    if (data.warning || (data.unknown_symptoms && data.unknown_symptoms.length > 0)) {
        warningSection = `
            <div class="warning-section">
                <div class="warning-header">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff9800" stroke-width="2">
                        <path d="M12 9v4m0 4h.01M12 3l9 18H3L12 3z"/>
                    </svg>
                    <h3>Symptom Validation Warning</h3>
                </div>
                <p class="warning-message">${data.warning || 'Some symptoms were not recognized'}</p>
                
                ${data.known_symptoms && data.known_symptoms.length > 0 ? `
                    <div class="symptom-validation-section">
                        <h4>✅ Recognized Symptoms (${data.known_symptoms.length}):</h4>
                        <div class="symptom-list">
                            ${data.known_symptoms.map(s => `<span class="validated-symptom known">${s}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${data.unknown_symptoms && data.unknown_symptoms.length > 0 ? `
                    <div class="symptom-validation-section">
                        <h4>❌ Unrecognized Symptoms (${data.unknown_symptoms.length}):</h4>
                        <div class="symptom-list">
                            ${data.unknown_symptoms.map(s => `<span class="validated-symptom unknown">${s}</span>`).join('')}
                        </div>
                        <p class="validation-note">These symptoms are not in our database and were not used for prediction.</p>
                    </div>
                ` : ''}
            </div>
        `;
    }

    // ===== XAI SECTION =====
    let xaiSection = '';
    if (data.xai_explanations && data.xai_explanations.length > 0) {
        const barColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];
        xaiSection = `
            <div class="detail-section xai-section">
                <h3>🧠 Why This Prediction? (AI Explanation)</h3>
                <p class="xai-subtitle">Feature importance analysis — how much each symptom contributed to this diagnosis:</p>
                <div class="xai-bars">
                    ${data.xai_explanations.map((item, i) => `
                        <div class="xai-bar-item">
                            <div class="xai-label">
                                <span class="xai-symptom">${item.symptom}</span>
                                <span class="xai-percentage">${item.percentage}%</span>
                            </div>
                            <div class="xai-bar-track">
                                <div class="xai-bar-fill" style="width: 0%; background: ${barColors[i % barColors.length]};" data-width="${item.percentage}%"></div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    // ===== TOP-3 PREDICTIONS =====
    let topPredictionsSection = '';
    if (data.top_predictions && data.top_predictions.length > 1) {
        topPredictionsSection = `
            <div class="detail-section top-predictions-section">
                <h3>📊 Top Predictions (Ensemble Analysis)</h3>
                <div class="top-predictions-list">
                    ${data.top_predictions.map((pred, i) => `
                        <div class="top-pred-item ${i === 0 ? 'top-pred-primary' : ''}">
                            <span class="top-pred-rank">#${i + 1}</span>
                            <span class="top-pred-name">${pred.disease}</span>
                            <div class="top-pred-bar-track">
                                <div class="top-pred-bar-fill" style="width: ${Math.min(pred.confidence, 100)}%; background: ${i === 0 ? '#17a2b8' : i === 1 ? '#6c757d' : '#adb5bd'}"></div>
                            </div>
                            <span class="top-pred-conf">${pred.confidence}%</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    // ===== RISK LEVEL BADGE =====
    const riskColors = { 'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e', 'Very Low': '#3b82f6' };
    const riskLevel = data.risk_level || 'Medium';

    // ===== TREATMENT SECTION =====
    let treatmentSection = '';
    if (data.treatment) {
        treatmentSection = `
            <div class="detail-section treatment-section">
                <h3>💊 Treatment Recommendations</h3>
                <p>${data.treatment}</p>
            </div>
        `;
    }

    // ===== ACTION PLAN SECTION =====
    let actionPlanSection = '';
    if (data.action_plan) {
        actionPlanSection = `
            <div class="detail-section action-plan-section">
                <h3>📋 Personalized Action Plan</h3>
                <p>${data.action_plan}</p>
            </div>
        `;
    }

    resultsBody.innerHTML = `
        <div class="result-header">
            <h2 class="disease-name">${data.disease}</h2>
            <div class="result-badges">
                <span class="confidence-badge" id="confidence-badge">${data.confidence}% Confidence</span>
                <span class="risk-badge" style="background: ${riskColors[riskLevel] || '#6c757d'}">${riskLevel} Risk</span>
            </div>
        </div>
        
        ${data.image_type ? `
            <div class="detected-type-badge" style="text-align:center; margin: 10px 0; padding: 6px 16px; background: #e8f4f8; border-radius: 8px; color: #17a2b8; font-weight: 600; display: inline-block; width: auto;">
                🔍 Detected from: ${data.image_type}
            </div>
        ` : ''}
        
        ${warningSection}
        ${xaiSection}
        ${topPredictionsSection}
        
        <div class="result-details">
            ${data.description ? `
                <div class="detail-section">
                    <h3>📋 Description</h3>
                    <p>${data.description}</p>
                </div>
            ` : ''}
            
            ${data.causes ? `
                <div class="detail-section">
                    <h3>🔍 Causes</h3>
                    <p>${data.causes}</p>
                </div>
            ` : ''}
            
            ${treatmentSection}
            ${actionPlanSection}
            
            ${data.prevention ? `
                <div class="detail-section">
                    <h3>🛡️ Prevention</h3>
                    <p>${data.prevention}</p>
                </div>
            ` : ''}
            
            ${data.doctors ? `
                <div class="detail-section">
                    <h3>👨‍⚕️ Suggested Doctors</h3>
                    <p>${data.doctors}</p>
                </div>
            ` : ''}
        </div>
    `;

    // Set confidence badge color
    const badge = document.getElementById('confidence-badge');
    if (data.confidence >= 80) {
        badge.style.background = '#28a745';
    } else if (data.confidence >= 60) {
        badge.style.background = '#ffc107';
    } else {
        badge.style.background = '#dc3545';
    }

    // Show modal
    const modal = document.getElementById('results-modal');
    modal.style.display = 'block';

    // Animate XAI bars after modal is visible
    setTimeout(() => {
        document.querySelectorAll('.xai-bar-fill').forEach(bar => {
            bar.style.width = bar.getAttribute('data-width');
        });
    }, 100);
}

// Close results modal
function closeResults() {
    const modal = document.getElementById('results-modal');
    modal.style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function (event) {
    const modal = document.getElementById('results-modal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}

// Show loading
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = show ? 'flex' : 'none';
    }
}

// Set button loading state with inline spinner
function setBtnLoading(btn, isLoading, text) {
    if (!btn) return;

    if (isLoading) {
        // Save original content so we can restore it later
        btn._originalHTML = btn.innerHTML;
        btn.innerHTML = '<div class="btn-spinner"></div> ' + text;
        btn.classList.add('loading');
        btn.disabled = true;
    } else {
        btn.innerHTML = btn._originalHTML || text;
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

// Show error
function showError(message) {
    alert('❌ ' + message);
}

// Reset form
function resetForm() {
    // Reset symptoms
    selectedSymptoms = [];
    customSymptoms = [];
    updateSelectedSymptoms();

    const symptomSearch = document.getElementById('symptom-search');
    if (symptomSearch) symptomSearch.value = '';

    const customInput = document.getElementById('custom-symptom-input');
    if (customInput) customInput.value = '';

    // Uncheck all checkboxes
    document.querySelectorAll('.symptom-checkbox input').forEach(checkbox => {
        checkbox.checked = false;
    });
    document.querySelectorAll('.symptom-checkbox').forEach(div => {
        div.classList.remove('selected');
    });

    // Reset image
    selectedFile = null;
    const imagePreview = document.getElementById('image-preview');
    if (imagePreview) {
        imagePreview.innerHTML = '';
        imagePreview.style.display = 'none';
    }

    const imageInput = document.getElementById('image-input');
    if (imageInput) imageInput.value = '';

    const analyzeBtn = document.getElementById('analyze-image-btn');
    if (analyzeBtn) {
        analyzeBtn.style.display = 'none';
        analyzeBtn.disabled = true;
    }



    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}