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
        console.log('Response result:', result);  // Debugging line to see the returned result

        if (result.error) {
            resultDiv.innerText = `Error occurred during detection: ${result.error}`;
        } else {
            const frameCount = result.frame_count;
            const detectionSummary = result.result;

            // Check if a weapon was detected and display the appropriate message
            const weaponDetected = detectionSummary.weapon_detected;
            const weaponName = detectionSummary.weapon_name || 'No weapon detected';

            resultDiv.innerHTML = `
                <p>Detection completed.</p>
                <p>Frames processed: ${frameCount}</p>
                <p>detection summary: ${weaponDetected ? weaponName : 'No weapon detected'}</p>
            `;
        }
    } catch (error) {
        resultDiv.innerText = `Error occurred during detection: ${error.message}`;
    } finally {
        loadingDiv.style.display = 'none';
    }
}
