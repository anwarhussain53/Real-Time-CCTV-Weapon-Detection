async function startDetection() {
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    
    if (fileInput.files.length === 0) {
        resultDiv.innerText = 'Please upload an image or video file.';
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    loadingDiv.style.display = 'block';
    resultDiv.innerText = '';

    try {
        const response = await fetch('http://localhost:5000/detect', { 
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        resultDiv.innerText = `Detection completed: ${result.result}`;
    } catch (error) {
        resultDiv.innerText = 'Error occurred during detection.';
    } finally {
        loadingDiv.style.display = 'none';
    }
}
