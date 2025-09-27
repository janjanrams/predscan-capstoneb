// Scroll to upload section function
function scrollToUpload() {
    const uploadSection = document.getElementById('upload-section');
    uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// PERFORMANCE OPTIMIZATION: Cache DOM elements and use efficient selectors
const DOMCache = {
    analysisSection: null,
    uploadForm: null,
    fileInput: null,
    uploadZone: null,
    fileInfo: null,
    errorMessage: null,
    fileName: null,
    fileSize: null,
    errorText: null,
    loadingBar: null,
    loadingProgress: null,
    loadingText: null,
    analyzeBtn: null
};

// Initialize DOM cache for better performance
function initDOMCache() {
    DOMCache.analysisSection = document.querySelector('.analysis-results-section');
    DOMCache.uploadForm = document.getElementById('upload-form');
    DOMCache.fileInput = document.getElementById('file');
    DOMCache.uploadZone = document.getElementById('upload-zone');
    DOMCache.fileInfo = document.getElementById('file-info');
    DOMCache.errorMessage = document.getElementById('error-message');
    DOMCache.fileName = document.getElementById('file-name');
    DOMCache.fileSize = document.getElementById('file-size');
    DOMCache.errorText = document.getElementById('error-text');
    DOMCache.loadingBar = document.getElementById('loading-bar');
    DOMCache.loadingProgress = document.getElementById('loading-progress');
    DOMCache.loadingText = document.getElementById('loading-text');
    DOMCache.analyzeBtn = document.querySelector('.analysis-btn');
}

// Debounced resize handler for better performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize DOM cache first for better performance
    initDOMCache();
    
    // Check if analysis results are present and stop loading if so
    if (DOMCache.analysisSection) {
        // Analysis results are present, stop the loading animation
        setTimeout(() => {
            if (typeof stopAnalysisLoading === 'function') {
                stopAnalysisLoading();
            }
        }, 500);
    }
    
    // Set up ResizeObserver for better responsive behavior
    if (window.ResizeObserver) {
        const analyzerRight = document.querySelector('.analyzer-right');
        if (analyzerRight) {
            const resizeObserver = new ResizeObserver(entries => {
                // Debounce the height matching update
                clearTimeout(window.heightMatchingTimeout);
                window.heightMatchingTimeout = setTimeout(updateHeightMatching, 100);
            });
            resizeObserver.observe(analyzerRight);
        }
    }

    // File input change handler (optimized with cached DOM elements)
    if (DOMCache.fileInput) {
        DOMCache.fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            
            if (!file) {
                resetUploadState();
                return;
            }

            // Validate file type
            if (file.type !== 'application/pdf') {
                showError('Please select a PDF file. Other file types are not supported.');
                return;
            }

            // Validate file size (10MB limit)
            const maxSize = 10 * 1024 * 1024;
            if (file.size > maxSize) {
                showError('File size exceeds 10MB limit. Please select a smaller file.');
                return;
            }

            // Show file info using cached elements
            if (DOMCache.fileName) DOMCache.fileName.textContent = file.name;
            if (DOMCache.fileSize) DOMCache.fileSize.textContent = formatFileSize(file.size);
            
            if (DOMCache.uploadZone) DOMCache.uploadZone.style.display = 'none';
            if (DOMCache.fileInfo) DOMCache.fileInfo.style.display = 'block';
            if (DOMCache.errorMessage) DOMCache.errorMessage.style.display = 'none';
            
            // Add success styling to upload zone
            if (DOMCache.uploadZone) DOMCache.uploadZone.classList.add('success');
            
            // Show document preview
            showDocumentPreview(URL.createObjectURL(file));
            
            // Add fade-in animation class for smooth appearance
            requestAnimationFrame(() => {
                const documentPreview = document.getElementById('document-preview');
                if (documentPreview) {
                    documentPreview.classList.add('fade-in');
                }
            });
        });
    }

    // Form submission handler (optimized with cached elements)
    if (DOMCache.uploadForm && DOMCache.fileInput) {
        DOMCache.uploadForm.addEventListener('submit', function(e) {
            const file = DOMCache.fileInput.files[0];
            
            if (!file) {
                e.preventDefault();
                showError('Please select a PDF file to analyze.');
                return;
            }

            // Start loading animation immediately for better UX
            startAnalysisLoading();
            
            // Optional: Add a small delay to ensure UI updates are visible
            setTimeout(() => {
                // Form will continue submitting naturally
            }, 50);
        });
    }

    // Drag and drop handlers (optimized with cached elements)
    if (DOMCache.uploadZone) {
        DOMCache.uploadZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            DOMCache.uploadZone.classList.add('drag-over');
        });

        DOMCache.uploadZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            DOMCache.uploadZone.classList.remove('drag-over');
        });

        DOMCache.uploadZone.addEventListener('drop', function(e) {
            e.preventDefault();
            DOMCache.uploadZone.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length === 1) {
                DOMCache.fileInput.files = files;
                DOMCache.fileInput.dispatchEvent(new Event('change'));
            } else {
                showError('Please drop only one PDF file.');
            }
        });
    }

    function resetUploadState() {
        // Use cached DOM elements for better performance
        if (DOMCache.uploadZone) {
            DOMCache.uploadZone.style.display = 'block';
            DOMCache.uploadZone.classList.remove('success');
        }
        if (DOMCache.fileInfo) DOMCache.fileInfo.style.display = 'none';
        if (DOMCache.errorMessage) DOMCache.errorMessage.style.display = 'none';
        if (DOMCache.fileInput) DOMCache.fileInput.value = '';
        
        // Hide document preview when resetting
        hideDocumentPreview();
    }

    function showError(message) {
        // Use cached DOM elements for better performance
        if (DOMCache.errorText) DOMCache.errorText.textContent = message;
        if (DOMCache.errorMessage) DOMCache.errorMessage.style.display = 'block';
        if (DOMCache.fileInfo) DOMCache.fileInfo.style.display = 'none';
        if (DOMCache.uploadZone) {
            DOMCache.uploadZone.style.display = 'block';
            DOMCache.uploadZone.classList.remove('success');
            DOMCache.uploadZone.classList.add('error');
            
            // Remove error class after animation using requestAnimationFrame
            setTimeout(() => {
                DOMCache.uploadZone.classList.remove('error');
            }, 500);
        }
        
        if (DOMCache.fileInput) DOMCache.fileInput.value = '';
        
        // Hide document preview on error
        hideDocumentPreview();
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Initialize responsive layout after DOM loads
    setTimeout(() => {
        updateHeightMatching();
        
        // Add intersection observer for analysis results to animate on appearance
        const analysisSection = document.querySelector('.analysis-results-section');
        if (analysisSection && 'IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('fade-in');
                    }
                });
            }, { threshold: 0.1 });
            
            observer.observe(analysisSection);
        }
    }, 100);
});

// Update height matching on window resize (optimized with debouncing)
const debouncedUpdateHeightMatching = debounce(updateHeightMatching, 150);
window.addEventListener('resize', debouncedUpdateHeightMatching);

// Document Preview Functions
function showDocumentPreview(pdfUrl) {
    const documentPreview = document.getElementById('document-preview');
    const previewFrame = document.getElementById('pdf-preview-frame');
    
    if (documentPreview && previewFrame) {
        console.log('Setting PDF preview URL:', pdfUrl);
        previewFrame.src = pdfUrl;
        documentPreview.style.display = 'block';
        
        // Add error handling for iframe
        previewFrame.onload = function() {
            console.log('PDF loaded successfully');
        };
        
        previewFrame.onerror = function() {
            console.error('Failed to load PDF:', pdfUrl);
            showPreviewError(pdfUrl);
        };
        
        // Add timeout fallback for cases where iframe doesn't trigger error event
        setTimeout(function() {
            try {
                // Check if iframe has loaded content successfully
                if (!previewFrame.contentDocument && !previewFrame.contentWindow) {
                    console.warn('PDF preview timeout - showing fallback');
                    showPreviewError(pdfUrl);
                }
            } catch (e) {
                // Cross-origin restrictions prevent access, which means PDF likely loaded successfully
                console.log('PDF preview loaded (cross-origin restrictions detected)');
            }
        }, 5000);
        
        // Add smooth entrance animation
        setTimeout(() => {
            documentPreview.classList.add('fade-in');
        }, 100);
    }
}

function hideDocumentPreview() {
    const documentPreview = document.getElementById('document-preview');
    const previewFrame = document.getElementById('pdf-preview-frame');
    
    if (documentPreview && previewFrame) {
        previewFrame.src = '';
        documentPreview.style.display = 'none';
        
        // Remove animation class when hiding
        documentPreview.classList.remove('fade-in');
    }
}

function showPreviewError(pdfUrl) {
    const previewFrame = document.getElementById('pdf-preview-frame');
    if (previewFrame) {
        // Show user-friendly error message in iframe
        previewFrame.srcdoc = `
            <div style="padding: 40px; text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); max-width: 400px;">
                    <div style="font-size: 48px; margin-bottom: 20px;">📄</div>
                    <h3 style="color: #374151; margin-bottom: 16px; font-size: 18px;">Preview Not Available</h3>
                    <p style="color: #6b7280; margin-bottom: 24px; line-height: 1.5;">Unable to display PDF preview in browser.</p>
                    <a href="${pdfUrl}" target="_blank" style="display: inline-block; background: #4f46e5; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; font-weight: 500; transition: background-color 0.2s;">
                        Open in New Tab
                    </a>
                </div>
            </div>
        `;
    }
}

function updateHeightMatching() {
    // Modern approach: Let CSS handle the layout naturally
    // This function ensures proper height constraints based on viewport
    
    const analyzerContainer = document.querySelector('.analyzer-container');
    const analyzerLeft = document.querySelector('.analyzer-left');
    const analyzerRight = document.querySelector('.analyzer-right');
    const featuresList = document.querySelector('.features-list');
    
    if (!featuresList || !analyzerContainer || !analyzerLeft || !analyzerRight) {
        return;
    }
    
    // For mobile screens (768px and below), ensure natural sizing
    if (window.innerWidth <= 768) {
        analyzerLeft.style.height = 'auto';
        analyzerLeft.style.maxHeight = 'none';
        analyzerLeft.style.overflow = 'visible';
        analyzerRight.style.height = 'auto';
        analyzerRight.style.maxHeight = 'none';
        analyzerRight.style.overflow = 'visible';
        featuresList.style.position = 'static';
        featuresList.style.height = 'auto';
        featuresList.style.maxHeight = '50vh';
        featuresList.style.minHeight = '300px';
        return;
    }
    
    // For larger screens, enable proper height matching
    if (window.innerWidth > 768) {
        analyzerLeft.style.height = '100%';
        analyzerLeft.style.maxHeight = '100%';
        analyzerLeft.style.overflow = 'hidden';
        analyzerRight.style.height = '100%';
        analyzerRight.style.maxHeight = '100%';
        analyzerRight.style.overflowY = 'auto';
        featuresList.style.position = 'static';
        featuresList.style.height = '100%';
        featuresList.style.maxHeight = '100%';
        featuresList.style.minHeight = '0';
    }
    
    // Trigger smooth transitions
    analyzerContainer.style.transition = 'all 0.3s ease';
}

// Loading Bar Functions (optimized with cached elements)
function startAnalysisLoading() {
    // Use cached DOM elements for better performance
    const loadingBar = DOMCache.loadingBar || document.getElementById('loading-bar');
    const loadingProgress = DOMCache.loadingProgress || document.getElementById('loading-progress');
    const loadingText = DOMCache.loadingText || document.getElementById('loading-text');
    const analyzeBtn = DOMCache.analyzeBtn || document.querySelector('.analysis-btn');
    
    if (!loadingBar || !loadingProgress || !loadingText || !analyzeBtn) {
        console.warn('Loading elements not found');
        return;
    }
    
    // Show loading elements
    loadingBar.style.display = 'block';
    loadingText.style.display = 'block';
    
    // Disable analyze button and show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span>Analyzing...';
    analyzeBtn.style.opacity = '0.7';
    
    // Reset progress bar
    loadingProgress.style.width = '0%';
    
    // Enhanced loading messages with more specific feedback
    const loadingMessages = [
        'Initializing analysis pipeline...',
        'Extracting and processing PDF content...',
        'Analyzing grammar with cached optimization...',
        'Computing lexical diversity metrics...',
        'Processing advanced linguistic features...',
        'Running machine learning prediction...',
        'Finalizing results and generating report...'
    ];
    
    // Optimized progress simulation
    let progress = 0;
    let messageIndex = 0;
    
    window.currentProgressInterval = setInterval(() => {
        progress += Math.random() * 8 + 4; // Slightly faster increment (4-12%)
        
        // Cap at 90% until analysis is complete
        if (progress >= 90) {
            progress = 90;
            clearInterval(window.currentProgressInterval);
        }
        
        // Update progress bar with smooth animation
        requestAnimationFrame(() => {
            loadingProgress.style.width = progress + '%';
        });
        
        // Update loading text based on progress with smoother transitions
        const newMessageIndex = Math.min(
            Math.floor((progress / 90) * (loadingMessages.length - 1)),
            loadingMessages.length - 1
        );
        
        if (newMessageIndex !== messageIndex) {
            messageIndex = newMessageIndex;
            loadingText.textContent = loadingMessages[messageIndex];
        }
        
    }, 250 + Math.random() * 200); // Optimized interval (250-450ms)
}

function stopAnalysisLoading() {
    // Use cached DOM elements for better performance
    const loadingBar = DOMCache.loadingBar || document.getElementById('loading-bar');
    const loadingProgress = DOMCache.loadingProgress || document.getElementById('loading-progress');
    const loadingText = DOMCache.loadingText || document.getElementById('loading-text');
    const analyzeBtn = DOMCache.analyzeBtn || document.querySelector('.analysis-btn');
    
    // Clear any existing interval
    if (window.currentProgressInterval) {
        clearInterval(window.currentProgressInterval);
    }
    
    if (loadingProgress && loadingText) {
        // Complete the progress bar to 100% with smooth animation
        requestAnimationFrame(() => {
            loadingProgress.style.width = '100%';
            loadingText.textContent = 'Analysis complete! 🎉';
        });
    }
    
    // Wait a moment to show 100% completion, then hide
    setTimeout(() => {
        // Hide loading elements
        if (loadingBar) loadingBar.style.display = 'none';
        if (loadingText) loadingText.style.display = 'none';
        
        // Reset analyze button
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span>Analyze Document';
            analyzeBtn.style.opacity = '1';
        }
    }, 1000); // Slightly longer to show success message
}

// Save Analysis function (legacy version)
function saveAnalysis() {
    // Get the save button and disable it during the request
    const saveBtn = document.querySelector('[onclick="saveAnalysis()"]');
    if (saveBtn) {
        saveBtn.disabled = true;
        saveBtn.style.opacity = '0.5';
        saveBtn.innerHTML = 'Saving...';
    }

    // Extract analysis data from DOM elements
    const analysisData = extractAnalysisDataFromDOM();
    
    // Send POST request to save analysis
    fetch('/dashboard/save-analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(analysisData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Save successful:', data);
        
        // Redirect to dashboard immediately
        window.location.href = '/dashboard/';
    })
    .catch(error => {
        // Show error message
        console.error('Error saving analysis:', error);
        alert('Error saving analysis to database. Please try again.');
    })
    .finally(() => {
        // Re-enable the save button
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.style.opacity = '1';
            saveBtn.innerHTML = 'Save Analysis';
        }
    });
}

// New save analysis function for updated format
function saveAnalysisNew() {
    // Get the save button and disable it during the request
    const saveBtn = document.querySelector('[onclick="saveAnalysisNew()"]');
    if (saveBtn) {
        saveBtn.disabled = true;
        saveBtn.style.opacity = '0.5';
        saveBtn.innerHTML = 'Saving...';
    }

    // Extract analysis data from the DOM elements
    const analysisData = extractAnalysisDataFromDOM();
    
    // Send POST request to save analysis
    fetch('/dashboard/save-analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(analysisData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(errorData => {
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        // Redirect immediately on successful save
        if (data.success) {
            // Show brief success message before redirect
            if (saveBtn) {
                saveBtn.innerHTML = '<span class="btn-icon">✓</span>Saved!';
                saveBtn.style.backgroundColor = '#10b981';
            }
            
            setTimeout(() => {
                window.location.href = '/dashboard/';
            }, 500);
        } else {
            throw new Error(data.message || 'Save operation failed');
        }
    })
    .catch(error => {
        // Show detailed error message
        console.error('Error saving analysis:', error);
        alert(`Error saving analysis: ${error.message || 'Unknown error occurred'}. Please try again.`);
    })
    .finally(() => {
        // Re-enable the save button
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.style.opacity = '1';
            saveBtn.innerHTML = '<span class="btn-icon">💾</span>Save Analysis';
        }
    });
}

// Helper function to extract analysis data from DOM
function extractAnalysisDataFromDOM() {
    const analysisData = {};
    
    try {
        // Check if analysis results section exists
        const analysisSection = document.querySelector('.analysis-results-section');
        if (!analysisSection) {
            throw new Error('No analysis result available to save');
        }

        // Extract data from DOM elements
        const predictionElement = document.querySelector('.result-title');
        const probabilityElements = document.querySelectorAll('.probability-metric-value');
        const confidenceElement = document.querySelector('.probability-metric-value.confidence');
        
        // Extract probabilities first to determine correct result
        let probability_legitimate = 50;
        let probability_predatory = 50;
        
        if (probabilityElements.length >= 2) {
            // Find the legitimate and predatory probability elements by their labels
            const probabilityColumns = document.querySelectorAll('.probability-column');
            for (const column of probabilityColumns) {
                const label = column.querySelector('.probability-metric-label');
                const value = column.querySelector('.probability-metric-value');
                
                if (label && value) {
                    const labelText = label.textContent.toLowerCase().trim();
                    const probabilityValue = parseFloat(value.textContent.replace('%', '')) || 50;
                    
                    if (labelText.includes('legitimate') || labelText.includes('non-predatory')) {
                        probability_legitimate = probabilityValue;
                    } else if (labelText.includes('predatory')) {
                        probability_predatory = probabilityValue;
                    }
                }
            }
        }
        
        analysisData.probability_legitimate = probability_legitimate;
        analysisData.probability_predatory = probability_predatory;
        
        // Determine result based on which probability is higher (matching the display logic)
        analysisData.result = probability_predatory > probability_legitimate ? 1 : 0;

        // Extract confidence
        analysisData.confidence = confidenceElement ? confidenceElement.textContent.trim() : 'Medium';

        // Extract feature values from the grid structure
        const extractFeatureFromGrid = (featureName) => {
            const gridRows = document.querySelectorAll('.grid-row');
            
            for (const row of gridRows) {
                const featureCell = row.querySelector('.feature-cell .feature-name');
                if (featureCell && featureCell.textContent.toLowerCase().includes(featureName.toLowerCase())) {
                    const analyzedCell = row.querySelector('.metric-cell.analyzed');
                    if (analyzedCell) {
                        const valueText = analyzedCell.textContent.trim();
                        
                        if (valueText === 'N/A') {
                            return 0;
                        }
                        // Extract numeric value, handling percentage signs and other formatting
                        const numericValue = valueText.replace(/[^\d.-]/g, '');
                        const finalValue = parseFloat(numericValue) || 0;
                        return finalValue;
                    }
                }
            }
            return 0;
        };

        // Extract individual features from the grid
        analysisData.review_speed = extractFeatureFromGrid('Review Speed');
        analysisData.word_count = extractFeatureFromGrid('Word Count');
        analysisData.lexical_density = extractFeatureFromGrid('Lexical Density') / 100; // Convert percentage to decimal
        analysisData.reference_count = extractFeatureFromGrid('Reference Count');
        analysisData.grammar_suggestions = extractFeatureFromGrid('Grammar Errors');
        
        // Set default values for features not displayed in the grid but expected by backend
        analysisData.hdd = 0.5; // Default HDD value
        analysisData.mtld = 50.0; // Default MTLD value

        // Set default description and remark based on result
        if (analysisData.result === 1) {
            analysisData.remark = 'Warning: Suspected Predatory';
            analysisData.description = 'The article was classified as Predatory by the model.';
        } else {
            analysisData.remark = 'Non-Predatory Publication Detected';
            analysisData.description = 'The article was classified as Non-Predatory by the model.';
        }

        // Get filename from file info if available
        const fileNameElement = document.getElementById('file-name');
        analysisData.filename = fileNameElement ? fileNameElement.textContent : 'uploaded_document.pdf';

        // Add user's local timestamp in local timezone
        const now = new Date();
        // Format as local datetime string that Python can parse
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        const milliseconds = String(now.getMilliseconds()).padStart(3, '0');
        
        // Create local time string in ISO-like format but representing local time
        analysisData.analysis_timestamp = `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.${milliseconds}`;

        return analysisData;
        
    } catch (error) {
        console.error('Error preparing analysis data:', error);
        throw error;
    }
}