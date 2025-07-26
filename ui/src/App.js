import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Camera, DollarSign, Loader, CheckCircle, AlertCircle } from 'lucide-react';
import './index.css';
  
function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setResult(null);
    } else {
      setError('Please select a valid image file.');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setError(null);
        setResult(null);
      } else {
        setError('Please select a valid image file.');
      }
    }
  };

  const handleUpload = () => {
    if (!selectedFile) {
      setError('Please select an image first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    axios.post(`${backendUrl}/convert-currency`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 30000, // 30 seconds timeout
    })
    .then(response => {
      setResult(response.data);
      setIsLoading(false);
    })
    .catch(err => {
      console.error('Error uploading image:', err);
      setError(err.response?.data?.message || 'Failed to convert currency. Please try again.');
      setIsLoading(false);
    });
  };

  const handleCameraCapture = () => {
    // This would integrate with device camera
    // For now, we'll just trigger the file input
    fileInputRef.current.click();
  };

  const resetForm = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>
            <DollarSign size={32} />
            Currency Converter
          </h1>
          <p>Upload a currency image to convert it to Nigerian Naira (₦)</p>
        </header>

        <div className="main-content">
          <div className="card form-card">
            {/* <div className="input-group">
              <label htmlFor="backend-url">Backend URL:</label>
              <input
                id="backend-url"
                type="text"
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                placeholder="http://localhost:8000"
              />
            </div> */}

            <div className="upload-section">
              <h3>Upload Currency Image</h3>
              
              <div
                className="upload-area"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current.click()}
              >
                <Upload size={48} color="#667eea" />
                <p>Drag and drop an image here, or click to browse</p>
                <p className="upload-hint">Supports: JPG, PNG, GIF, WebP</p>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />

              <div className="upload-actions">
                <button className="btn btn-secondary" onClick={handleCameraCapture}>
                  <Camera size={20} />
                  Take Photo
                </button>
              </div>

              {previewUrl && (
                <div className="preview-section">
                  <h4>Preview:</h4>
                  <img src={previewUrl} alt="Preview" className="preview-image" />
                  <div className="file-info">
                    <p><strong>File:</strong> {selectedFile?.name}</p>
                    <p><strong>Size:</strong> {(selectedFile?.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
              )}

              {error && (
                <div className="error">
                  <AlertCircle size={20} />
                  {error}
                </div>
              )}

              <div className="action-buttons">
                <button 
                  className="btn" 
                  onClick={handleUpload}
                  disabled={!selectedFile || isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader size={20} className="spinner" />
                      Converting...
                    </>
                  ) : (
                    <>
                      <DollarSign size={20} />
                      Convert to Naira
                    </>
                  )}
                </button>
                
                {selectedFile && (
                  <button className="btn btn-secondary" onClick={resetForm}>
                    Reset
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="side-result">
            {isLoading && (
              <div className="card">
                <div className="loading">
                  <div className="spinner"></div>
                  <p>Processing your currency image...</p>
                </div>
              </div>
            )}

            {result && (
              <div className="card result-card">
                <CheckCircle size={48} />
                <h2>Conversion Result</h2>
                <div className="result-details">
                  <p><strong>Original Currency:</strong> {result.originalCurrency || 'Unknown'}</p>
                  <p><strong>Original Amount:</strong> {result.originalAmount || 'Unknown'}</p>
                  <p><strong>Converted Amount:</strong> ₦{result.convertedAmount || 'Unknown'}</p>
                  <p><strong>Exchange Rate:</strong> {result.exchangeRate || 'Unknown'}</p>
                  {result.confidence && (
                    <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
                  )}
                </div>
                <button className="btn btn-secondary" onClick={resetForm}>
                  Convert Another Image
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 