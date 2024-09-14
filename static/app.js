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
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.error) {
            resultDiv.innerText = `Error occurred during detection: ${result.error}`;
        } else {
            // Summarize detection results
            const summary = result.result.flat().reduce((acc, detection) => {
                acc[detection] = (acc[detection] || 0) + 1;
                return acc;
            }, {});

            // Generate HTML for the result
            const summaryHtml = Object.entries(summary).map(([label, count]) => `
                <li>${label}: ${count}</li>
            `).join('');

            resultDiv.innerHTML = `
                <p>Detection completed.</p>
                <p>Frames processed: ${result.frame_count}</p>
                <p>Detection summary:</p>
                <ul>
                    ${summaryHtml}
                </ul>
            `;
        }
    } catch (error) {
        resultDiv.innerText = `Error occurred during detection: ${error.message}`;
    } finally {
        loadingDiv.style.display = 'none';
    }
}
