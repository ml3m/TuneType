document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileUpload = document.getElementById('file-upload');
    const selectedFileContainer = document.getElementById('selected-file');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeFileBtn = document.getElementById('remove-file');
    const submitButton = document.getElementById('submit-button');
    const loadingOverlay = document.getElementById('loading-overlay');
    const uploadContainer = document.getElementById('upload-container');
    const resultsContainer = document.getElementById('results-container');
    const topGenre = document.getElementById('top-genre');
    const topConfidence = document.getElementById('top-confidence');
    const genreBars = document.getElementById('genre-bars');
    const backButton = document.getElementById('back-button');
    const visualizeButton = document.getElementById('visualize-button');
    
    let selectedFile = null;
    let currentFileName = null;
    
    // Handle file selection
    function handleFileSelect(file) {
        if (!file) return;
        
        // Check if file is an allowed audio format
        const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/x-wav'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const allowedExtensions = ['mp3', 'wav', 'ogg'];
        
        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
            alert('Please select an MP3, WAV, or OGG file.');
            return;
        }
        
        // Check file size (max 50MB)
        if (file.size > 50 * 1024 * 1024) {
            alert('File size exceeds 50MB limit.');
            return;
        }
        
        selectedFile = file;
        updateFileInfo();
    }
    
    function updateFileInfo() {
        if (selectedFile) {
            fileName.textContent = selectedFile.name;
            fileSize.textContent = formatFileSize(selectedFile.size);
            selectedFileContainer.style.display = 'flex';
            submitButton.disabled = false;
        } else {
            selectedFileContainer.style.display = 'none';
            submitButton.disabled = true;
        }
    }
    
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
        else return (bytes / 1048576).toFixed(2) + ' MB';
    }
    
    // Event Listeners
    uploadArea.addEventListener('click', function() {
        fileUpload.click();
    });
    
    fileUpload.addEventListener('change', function(e) {
        handleFileSelect(e.target.files[0]);
    });
    
    // Drag and Drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }
    
    uploadArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFileSelect(file);
    });
    
    // Remove file button
    removeFileBtn.addEventListener('click', function() {
        selectedFile = null;
        fileUpload.value = '';
        updateFileInfo();
    });
    
    // Submit button - Upload file and analyze
    submitButton.addEventListener('click', function() {
        if (!selectedFile) return;
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Send to server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Server error');
                });
            }
            return response.json();
        })
        .then(data => {
            currentFileName = data.filename;
            displayResults(data);
        })
        .catch(error => {
            alert('Error: ' + error.message);
            loadingOverlay.style.display = 'none';
        });
    });
    
    // Visualize button
    if (visualizeButton) {
        visualizeButton.addEventListener('click', function() {
            if (currentFileName) {
                window.open(`/visualize/${currentFileName}`, '_blank');
            }
        });
    }
    
    // Display results
    function displayResults(data) {
        // Hide loading overlay
        loadingOverlay.style.display = 'none';
        
        // Hide upload container and show results
        uploadContainer.style.display = 'none';
        resultsContainer.style.display = 'block';
        
        // Update UI with results
        topGenre.textContent = capitalizeFirstLetter(data.top_genre);
        topConfidence.textContent = Math.round(data.top_probability * 100) + '%';
        
        // Clear previous genre bars
        genreBars.innerHTML = '';
        
        // Add genre bars for top 6 genres
        const maxDisplayGenres = Math.min(6, data.results.length);
        
        for (let i = 0; i < maxDisplayGenres; i++) {
            const result = data.results[i];
            const genre = result.genre;
            const probability = result.probability;
            const percentage = Math.round(probability * 100);
            
            const genreBar = document.createElement('div');
            genreBar.className = 'genre-bar';
            
            genreBar.innerHTML = `
                <div class="genre-bar-label">
                    <span class="genre-name">${capitalizeFirstLetter(genre)}</span>
                    <span class="genre-percentage">${percentage}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${percentage}%"></div>
                </div>
            `;
            
            genreBars.appendChild(genreBar);
        }
    }
    
    // Back button - Reset and return to upload page
    backButton.addEventListener('click', function() {
        // Hide results and show upload container
        resultsContainer.style.display = 'none';
        uploadContainer.style.display = 'block';
        
        // Reset file selection
        selectedFile = null;
        fileUpload.value = '';
        updateFileInfo();
    });
    
    // Helper function to capitalize first letter
    function capitalizeFirstLetter(string) {
        if (!string) return '';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
}); 