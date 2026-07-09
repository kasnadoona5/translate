// Tarjomeh Web App - Client Logic and SSE Streaming

let selectedFile = null;
let currentEventSource = null;
let currentReviewJobId = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    setupDragAndDrop();
    fetchJobs();
    fetchGlossaryTerms();
    
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

    let jobFinished = false;

    const finish = (stage) => {
        jobFinished = true;
        if (currentEventSource) {
            currentEventSource.close();
            currentEventSource = null;
        }
        const ok = stage === "complete";
        const paused = stage === "paused";
        appendLog(`Job finished: ${ok ? "COMPLETED ✅" : stage.toUpperCase()}`, ok ? "success" : (paused ? "warning" : "error"));
        setTimeout(() => {
            document.getElementById("uploadSection").style.display = "block";
            document.getElementById("progressSection").style.display = "none";
            document.getElementById("startBtn").disabled = false;
            document.getElementById("startBtn").innerText = "🚀 Start Translation";
            fetchJobs();
        }, 3000);
    };

    currentEventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            // Ignore keepalives and stream-closed notices (no real content).
            if (data.stage === "keepalive") return;
            if (data.stage === "closed") { if (!jobFinished) finish("complete"); return; }

            // Progress: the server sends `progress` (0.0–1.0).
            if (typeof data.progress === "number") {
                const pct = Math.round(data.progress * 100);
                progressBar.style.width = `${pct}%`;
                progressPct.innerText = `${pct}%`;
            }
            if (data.message) {
                progressStage.innerText = `${data.stage}: ${data.message}`;
            }

            let logClass = "info";
            if (data.stage === "error") logClass = "error";
            else if (data.stage === "complete" || data.stage === "Typography") logClass = "success";
            appendLog(`[${data.stage}] ${data.message || ""}`, logClass);

            // Terminal events use `stage` (the server never sends `status` here).
            if (data.stage === "complete" || data.stage === "error" || data.stage === "paused") {
                finish(data.stage);
            }
        } catch (e) {
            console.error("Failed to parse SSE packet", e);
        }
    };

    currentEventSource.onerror = () => {
        // After completion the server closes the stream — that's expected, not
        // an error. Only warn (and let EventSource retry) while still running.
        if (jobFinished) {
            if (currentEventSource) { currentEventSource.close(); currentEventSource = null; }
            return;
        }
        appendLog("Reconnecting to server…", "warning");
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

            if (job.status === "completed") {
                actionHtml = `
                    <button class="action-btn" title="Review QA" onclick="openReview('${job.id}')">QA</button>
                    <button class="action-btn" title="Download report" onclick="downloadQaReport('${job.id}')">Report</button>
                    ${actionHtml}
                `;
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

function authUrl(path) {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    return token ? `${path}${path.includes("?") ? "&" : "?"}token=${token}` : path;
}

async function openReview(jobId) {
    currentReviewJobId = jobId;
    const section = document.getElementById("reviewSection");
    const list = document.getElementById("reviewList");
    document.getElementById("reviewJobTitle").innerText = `Job ${jobId}`;
    section.style.display = "block";
    list.innerHTML = `<p class="empty-state">Loading review data...</p>`;

    try {
        const response = await fetch(authUrl(`/api/jobs/${jobId}/review`));
        if (!response.ok) throw new Error("Failed to load review data");
        const data = await response.json();
        const chunks = data.chunks || [];
        list.innerHTML = "";

        chunks.forEach(chunk => {
            const div = document.createElement("div");
            div.className = `review-item ${chunk.flagged ? "flagged" : ""}`;
            const score = chunk.critique_average === null || chunk.critique_average === undefined
                ? "n/a"
                : Number(chunk.critique_average).toFixed(1);
            div.innerHTML = `
                <div class="review-head">
                    <strong>Chunk ${chunk.chunk_index}</strong>
                    <span>Status: ${chunk.status}</span>
                    <span>Critique: ${score}</span>
                    <span>Glossary: ${chunk.glossary_violations}</span>
                    <span>Back-check: ${chunk.back_translation_flagged ? "flagged" : "ok/unsampled"}</span>
                    <button class="btn" onclick="retranslateChunk('${jobId}', ${chunk.chunk_index})">Retranslate</button>
                </div>
                <div class="review-columns">
                    <pre>${escapeHtml(chunk.source || "")}</pre>
                    <pre dir="rtl">${escapeHtml(chunk.translation || "")}</pre>
                </div>
            `;
            list.appendChild(div);
        });
    } catch (e) {
        list.innerHTML = `<p class="empty-state error">${e.message}</p>`;
    }
}

async function retranslateChunk(jobId, chunkIndex) {
    if (!confirm(`Retranslate chunk ${chunkIndex}? This will call the LLM again for this chunk.`)) return;
    const response = await fetch(authUrl(`/api/jobs/${jobId}/chunks/${chunkIndex}/retranslate`), { method: "POST" });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        alert(data.error || "Retranslation failed");
        return;
    }
    await openReview(jobId);
    fetchJobs();
}

async function exportReviewedJob() {
    if (!currentReviewJobId) return;
    const fmt = document.getElementById("exportFormat").value;
    const response = await fetch(authUrl(`/api/jobs/${currentReviewJobId}/export`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ format: fmt })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        alert(data.error || "Export failed");
        return;
    }
    alert(`Exported: ${data.output_path}`);
    fetchJobs();
}

function downloadQaReport(jobId = null) {
    const id = jobId || currentReviewJobId;
    if (!id) return;
    window.open(authUrl(`/api/jobs/${id}/qa-report`), "_blank");
}

async function fetchGlossaryTerms() {
    const list = document.getElementById("glossaryList");
    if (!list) return;
    try {
        const response = await fetch(authUrl("/api/glossary/terms"));
        if (!response.ok) throw new Error("Failed to load glossary");
        const data = await response.json();
        const terms = data.terms || [];
        list.innerHTML = "";
        terms.slice(0, 80).forEach(term => {
            const row = document.createElement("div");
            row.className = `glossary-row ${term.is_auto ? "auto" : ""}`;
            row.innerHTML = `
                <span><strong>${escapeHtml(term.source)}</strong> -> ${escapeHtml(term.target)}</span>
                <span>${escapeHtml(term.domain || "")}</span>
                <button class="action-btn" onclick="deleteGlossaryTerm(${term.index})">Delete</button>
                ${term.is_auto ? `<button class="action-btn" onclick="approveGlossaryTerm(${term.index})">Approve</button>` : ""}
            `;
            list.appendChild(row);
        });
    } catch (e) {
        list.innerHTML = `<p class="empty-state error">${e.message}</p>`;
    }
}

async function addGlossaryTerm() {
    const payload = {
        source: document.getElementById("glossSource").value,
        target: document.getElementById("glossTarget").value,
        domain: document.getElementById("glossDomain").value,
        context: document.getElementById("glossContext").value
    };
    const response = await fetch(authUrl("/api/glossary/terms"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        alert(data.error || "Failed to add term");
        return;
    }
    ["glossSource", "glossTarget", "glossDomain", "glossContext"].forEach(id => document.getElementById(id).value = "");
    fetchGlossaryTerms();
}

async function deleteGlossaryTerm(index) {
    if (!confirm("Delete this glossary term?")) return;
    await fetch(authUrl(`/api/glossary/terms/${index}`), { method: "DELETE" });
    fetchGlossaryTerms();
}

async function approveGlossaryTerm(index) {
    await fetch(authUrl(`/api/glossary/terms/${index}/approve`), { method: "POST" });
    fetchGlossaryTerms();
}

function downloadGlossary() {
    window.open(authUrl("/api/glossary/download"), "_blank");
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
