// Tarjomeh Web App - Client Logic and SSE Streaming

let selectedFile = null;
let currentEventSource = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    setupDragAndDrop();
    fetchJobs();
    
    // File input change handler
    document.getElementById("fileInput").addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
});

// Setup Drag & Drop Zone handlers
function setupDragAndDrop() {
    const dropZone = document.getElementById("dropZone");
    
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
}

// Handle selected file details
function handleFileSelect(file) {
    selectedFile = file;
    document.getElementById("selectedFileName").innerText = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    document.getElementById("configPanel").style.display = "flex";
}

// Start translation pipeline job
async function startTranslation() {
    if (!selectedFile) return;

    const startBtn = document.getElementById("startBtn");
    startBtn.disabled = true;
    startBtn.innerText = "Uploading...";

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("mode", document.getElementById("cfgMode").value);
    formData.append("format", document.getElementById("cfgFormat").value);
    formData.append("bilingual_mode", document.getElementById("cfgBilingual").value);

    // Get Auth token if set
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    let url = "/api/translate";
    if (token) {
        url += `?token=${token}`;
    }

    try {
        const response = await fetch(url, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Upload failed");
        }

        const data = await response.json();
        const jobId = data.job_id;
        
        // Hide upload form and show progress window
        document.getElementById("uploadSection").style.display = "none";
        document.getElementById("progressSection").style.display = "block";
        
        trackJobProgress(jobId);

    } catch (err) {
        alert(`Error: ${err.message}`);
        startBtn.disabled = false;
        startBtn.innerText = "🚀 Start Translation";
    }
}

// Connect to SSE stream for real-time logs
function trackJobProgress(jobId) {
    if (currentEventSource) {
        currentEventSource.close();
    }

    const logContainer = document.getElementById("progressLog");
    const progressBar = document.getElementById("progressBar");
    const progressPct = document.getElementById("progressPct");
    const progressStage = document.getElementById("progressStage");
    
    // Clear logs
    logContainer.innerHTML = "";
    progressBar.style.width = "0%";
    progressPct.innerText = "0%";
    progressStage.innerText = "Initializing...";

    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    let streamUrl = `/api/jobs/${jobId}/stream`;
    if (token) {
        streamUrl += `?token=${token}`;
    }

    currentEventSource = new EventSource(streamUrl);
    appendLog("Connecting to translation pipeline stream...", "info");

    currentEventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            // Update progress bar
            const pct = Math.round(data.pct * 100);
            progressBar.style.width = `${pct}%`;
            progressPct.innerText = `${pct}%`;
            progressStage.innerText = `${data.stage}: ${data.message}`;

            let logClass = "info";
            if (data.stage === "Error") logClass = "error";
            else if (data.stage === "Complete") logClass = "success";
            else if (data.stage === "Typography") logClass = "success";
            
            appendLog(`[${data.stage}] ${data.message}`, logClass);

            // Check final statuses
            if (data.status === "completed" || data.status === "failed" || data.status === "paused_error") {
                currentEventSource.close();
                currentEventSource = null;
                appendLog(`Job execution finished with status: ${data.status.toUpperCase()}`, data.status === "completed" ? "success" : "error");
                
                // Show action buttons or reset state
                setTimeout(() => {
                    document.getElementById("uploadSection").style.display = "block";
                    document.getElementById("progressSection").style.display = "none";
                    document.getElementById("startBtn").disabled = false;
                    document.getElementById("startBtn").innerText = "🚀 Start Translation";
                    fetchJobs();
                }, 4000);
            }

        } catch (e) {
            console.error("Failed to parse SSE packet", e);
        }
    };

    currentEventSource.onerror = (err) => {
        console.error("SSE connection error", err);
        appendLog("Disconnected from server. Retrying connection...", "warning");
    };
}

// Log utility
function appendLog(message, type = "info") {
    const logContainer = document.getElementById("progressLog");
    const div = document.createElement("div");
    div.className = `log-item ${type}`;
    div.innerText = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContainer.appendChild(div);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// Fetch all jobs in history
async function fetchJobs() {
    const listContainer = document.getElementById("jobsList");
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    let url = "/api/jobs";
    if (token) {
        url += `?token=${token}`;
    }

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error("Failed to load jobs");
        
        const resData = await response.json();
        const jobs = resData.jobs || [];
        listContainer.innerHTML = "";
        
        if (jobs.length === 0) {
            listContainer.innerHTML = `<p class="empty-state">No translation history found.</p>`;
            return;
        }

        jobs.forEach(job => {
            const div = document.createElement("div");
            div.className = "job-item";
            
            // Format status badge
            let badgeClass = "processing";
            if (job.status === "completed") badgeClass = "completed";
            else if (job.status === "failed") badgeClass = "failed";
            else if (job.status === "paused_error") badgeClass = "failed";
            else if (job.status === "paused") badgeClass = "paused";

            const progressPct = Math.round(job.pct * 100);

            // Action buttons
            let actionHtml = "";
            if (job.status === "completed") {
                actionHtml = `<button class="action-btn" title="Download output" onclick="downloadJob('${job.id}')">💾</button>`;
            } else if (job.status === "processing") {
                actionHtml = `
                    <button class="action-btn" title="Track Progress" onclick="trackJobProgress('${job.id}')">👁️</button>
                    <button class="action-btn" title="Pause" onclick="pauseJob('${job.id}')">⏸️</button>
                `;
            } else if (job.status === "paused" || job.status === "paused_error") {
                actionHtml = `<button class="action-btn" title="Resume" onclick="resumeJob('${job.id}')">▶️</button>`;
            }

            div.innerHTML = `
                <div class="job-meta">
                    <span class="job-title">${job.filename}</span>
                    <span class="job-submeta">ID: ${job.id} | Mode: ${job.mode.toUpperCase()} | Progress: ${progressPct}%</span>
                </div>
                <div class="job-actions">
                    <span class="badge ${badgeClass}">${job.status.toUpperCase()}</span>
                    ${actionHtml}
                </div>
            `;
            listContainer.appendChild(div);
        });

    } catch (e) {
        listContainer.innerHTML = `<p class="empty-state error">Error loading jobs history: ${e.message}</p>`;
    }
}

// Download output document
function downloadJob(jobId) {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    let url = `/api/jobs/${jobId}/download`;
    if (token) {
        url += `?token=${token}`;
    }
    window.open(url, "_blank");
}

// Pause job
async function pauseJob(jobId) {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    let url = `/api/jobs/${jobId}/pause`;
    if (token) {
        url += `?token=${token}`;
    }

    try {
        await fetch(url, { method: "POST" });
        fetchJobs();
    } catch (e) {
        alert("Failed to pause job");
    }
}

// Resume job
async function resumeJob(jobId) {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    let url = `/api/jobs/${jobId}/resume`;
    if (token) {
        url += `?token=${token}`;
    }

    try {
        await fetch(url, { method: "POST" });
        document.getElementById("uploadSection").style.display = "none";
        document.getElementById("progressSection").style.display = "block";
        trackJobProgress(jobId);
    } catch (e) {
        alert("Failed to resume job");
    }
}
