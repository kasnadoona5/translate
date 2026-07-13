// Tarjomeh Web App - Client Logic and SSE Streaming

let selectedFile = null;
let currentEventSource = null;
let currentReviewJobId = null;
let comparisonBaselineJobId = null;
let currentEvaluationId = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    setupDragAndDrop();
    setupSettingsControls();
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
    document.getElementById("selectedFileName").innerText = file.name;
    document.getElementById("selectedFileMeta").innerText =
        formatFileSize(file.size) + " · " +
        file.name.split(".").pop().toUpperCase();
    document.querySelector(".file-indicator").innerText =
        file.name.split(".").pop().slice(0, 3).toUpperCase();
    document.getElementById("configPanel").hidden = false;
    updateRunSummary();
}

function setupSettingsControls() {
    const mode = document.getElementById("cfgMode");
    const format = document.getElementById("cfgFormat");
    const bilingual = document.getElementById("cfgBilingual");
    const termNotes = document.getElementById("cfgTermNotes");

    mode.addEventListener("change", () => {
        const presets = {
            fast: {
                critique: false, backTranslation: false, webContext: false,
                refinements: 0, threshold: 9, sample: 0
            },
            quality: {
                critique: true, backTranslation: true, webContext: true,
                refinements: 1, threshold: 9, sample: 5
            },
            academic: {
                critique: true, backTranslation: true, webContext: true,
                refinements: 2, threshold: 9, sample: 20
            }
        };
        const preset = presets[mode.value];
        document.getElementById("cfgCritique").checked = preset.critique;
        document.getElementById("cfgBackTranslation").checked = preset.backTranslation;
        document.getElementById("cfgWebContext").checked = preset.webContext;
        document.getElementById("cfgRefineIterations").value = preset.refinements;
        document.getElementById("cfgCritiqueThreshold").value = preset.threshold;
        document.getElementById("cfgBackSample").value = preset.sample;
        updateRunSummary();
    });

    format.addEventListener("change", () => {
        syncTermNoteAvailability();
        updateRunSummary();
    });
    bilingual.addEventListener("change", updateRunSummary);
    termNotes.addEventListener("change", updateRunSummary);
    syncTermNoteAvailability();
    updateRunSummary();
}

function syncTermNoteAvailability() {
    const format = document.getElementById("cfgFormat").value;
    const termNotes = document.getElementById("cfgTermNotes");
    const supported = ["docx", "epub", "markdown"].includes(format);
    if (!supported) termNotes.value = "inline";
    termNotes.disabled = !supported;
}

function updateRunSummary() {
    const summary = document.getElementById("runSummary");
    if (!summary) return;
    const mode = document.getElementById("cfgMode");
    const format = document.getElementById("cfgFormat");
    const bilingual = document.getElementById("cfgBilingual");
    summary.innerText = [
        mode.options[mode.selectedIndex].text,
        format.options[format.selectedIndex].text,
        bilingual.options[bilingual.selectedIndex].text
    ].join(" · ");
}

function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) {
        return Math.max(1, Math.round(bytes / 1024)) + " KB";
    }
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
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
    formData.append("term_notes", document.getElementById("cfgTermNotes").value);
    formData.append("enable_book_research", String(document.getElementById("cfgBookResearch").checked));
    formData.append("enable_critique", String(document.getElementById("cfgCritique").checked));
    formData.append("enable_back_translation", String(document.getElementById("cfgBackTranslation").checked));
    formData.append("enable_integrity_gate", String(document.getElementById("cfgIntegrityGate").checked));
    formData.append("enable_web_context", String(document.getElementById("cfgWebContext").checked));
    formData.append("enable_auto_extraction", String(document.getElementById("cfgAutoExtraction").checked));
    formData.append("enable_compliance_check", String(document.getElementById("cfgCompliance").checked));
    formData.append("enable_auto_correction", String(document.getElementById("cfgAutoCorrection").checked));
    formData.append("scholarly_mode", String(document.getElementById("cfgScholarly").checked));
    formData.append("max_refine_iterations", document.getElementById("cfgRefineIterations").value);
    formData.append("critique_threshold", document.getElementById("cfgCritiqueThreshold").value);
    formData.append("qa_json_retries", document.getElementById("cfgQaJsonRetries").value);
    formData.append("recovery_model", document.getElementById("cfgRecoveryModel").value.trim());
    formData.append("back_translation_sample_pct", document.getElementById("cfgBackSample").value);
    formData.append("search_provider", document.getElementById("cfgSearchProvider").value);
    formData.append("phase7_max_queries", document.getElementById("cfgResearchQueries").value);
    formData.append("max_queries_per_chunk", document.getElementById("cfgChunkQueries").value);
    formData.append("max_queries_per_book", document.getElementById("cfgBookQueryBudget").value);

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
        document.getElementById("uploadSection").hidden = true;
        document.getElementById("progressSection").hidden = false;
        
        trackJobProgress(jobId);

    } catch (err) {
        alert(`Error: ${err.message}`);
        startBtn.disabled = false;
        startBtn.innerHTML = "Start translation <span aria-hidden='true'>→</span>";
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
            document.getElementById("uploadSection").hidden = false;
            document.getElementById("progressSection").hidden = true;
            document.getElementById("startBtn").disabled = false;
            document.getElementById("startBtn").innerHTML =
                "Start translation <span aria-hidden='true'>→</span>";
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

function formatJobDate(value) {
    if (!value) return "Unknown time";
    const normalized = /(?:Z|[+-]\d\d:\d\d)$/.test(value)
        ? value : value + "Z";
    const date = new Date(normalized);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString(undefined, {
        dateStyle: "medium", timeStyle: "short"
    });
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
            const outputName = job.output_filename || job.filename;
            const outputFormat = (job.output_format || "unknown").toUpperCase();
            const createdAt = formatJobDate(job.created_at);

            // Action buttons
            let actionHtml = "";
            if (job.status === "completed") {
                actionHtml = `<button class="action-btn" title="Download output" onclick="downloadJob('${job.id}')">Download</button>`;
            } else if (job.status === "processing") {
                actionHtml = `
                    <button class="action-btn" title="Track progress" onclick="trackJobProgress('${job.id}')">Track</button>
                    <button class="action-btn" title="Pause job" onclick="pauseJob('${job.id}')">Pause</button>
                `;
            } else if (job.status === "paused" || job.status === "paused_error") {
                actionHtml = `<button class="action-btn" title="Resume job" onclick="resumeJob('${job.id}')">Resume</button>`;
            }

            if (job.status === "completed") {
                actionHtml = `
                    <button class="action-btn" title="Use this job in a quality comparison" onclick="selectComparisonJob('${job.id}')">Compare</button>
                    <button class="action-btn" title="Review translation" onclick="openReview('${job.id}')">Review</button>
                    <button class="action-btn" title="Download QA report" onclick="downloadQaReport('${job.id}')">QA report</button>
                    ${actionHtml}
                `;
            }

            div.innerHTML = `
                <div class="job-meta">
                    <span class="job-title">${escapeHtml(outputName)}</span>
                    <span class="job-submeta">Source: ${escapeHtml(job.filename)} | Format: ${escapeHtml(outputFormat)}</span>
                    <span class="job-submeta">Created: ${escapeHtml(createdAt)} | ID: ${job.id} | Mode: ${job.mode.toUpperCase()} | Progress: ${progressPct}%</span>
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
        document.getElementById("uploadSection").hidden = true;
        document.getElementById("progressSection").hidden = false;
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

async function selectComparisonJob(jobId) {
    const section = document.getElementById("evaluationSection");
    const status = document.getElementById("evaluationStatus");
    section.hidden = false;
    if (!comparisonBaselineJobId) {
        comparisonBaselineJobId = jobId;
        currentEvaluationId = null;
        document.getElementById("evaluationSummary").innerHTML = "";
        document.getElementById("evaluationList").innerHTML = "";
        document.getElementById("evaluationDownloads").hidden = true;
        status.innerText = `Baseline selected: ${jobId}. Choose Compare on a second completed job.`;
        section.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
    }
    if (comparisonBaselineJobId === jobId) {
        status.innerText = "Choose a different completed job as the candidate.";
        return;
    }

    const baseline = comparisonBaselineJobId;
    comparisonBaselineJobId = null;
    status.innerText = "Running local integrity checks...";
    const response = await fetch(authUrl("/api/evaluations"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ baseline_job_id: baseline, candidate_job_id: jobId })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        status.innerText = data.error || "Evaluation failed.";
        return;
    }
    await openEvaluation(data.evaluation_id);
}

async function openEvaluation(evaluationId) {
    currentEvaluationId = evaluationId;
    const response = await fetch(authUrl(`/api/evaluations/${evaluationId}`));
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "Failed to load evaluation");
    document.getElementById("evaluationStatus").innerText =
        `Evaluation ${evaluationId} | ${formatJobDate(data.created_at)}`;
    document.getElementById("evaluationDownloads").hidden = false;
    const summary = data.summary || {};
    document.getElementById("evaluationSummary").innerHTML = `
        <span class="metric">${summary.chunks_compared || 0} chunks</span>
        <span class="metric">${summary.integrity_regressions || 0} regressions</span>
        <span class="metric">${summary.integrity_improvements || 0} improvements</span>
        <span class="metric ${summary.release_blocked ? "metric-alert" : "metric-ok"}">
            ${summary.release_blocked ? "Release check blocked" : "Integrity check clear"}
        </span>`;

    const list = document.getElementById("evaluationList");
    list.innerHTML = "";
    (data.chunks || []).forEach(chunk => {
        const item = document.createElement("div");
        item.className = "evaluation-item";
        const saved = chunk.preference && chunk.preference.preference
            ? `Saved: ${chunk.preference.preference}` : "Not reviewed";
        item.innerHTML = `
            <div class="review-head">
                <strong>Chunk ${chunk.chunk_index}</strong>
                <span>${chunk.source_aligned ? "Source aligned" : "Source mismatch"}</span>
                <span>${saved}</span>
            </div>
            <details class="evaluation-source">
                <summary>Source text</summary>
                <pre>${escapeHtml(chunk.source || "")}</pre>
            </details>
            <div class="evaluation-columns">
                <div><strong>Translation A</strong><pre dir="rtl">${escapeHtml(chunk.translation_a || "")}</pre></div>
                <div><strong>Translation B</strong><pre dir="rtl">${escapeHtml(chunk.translation_b || "")}</pre></div>
            </div>
            <div class="evaluation-actions">
                <button class="action-btn" onclick="saveEvaluationPreference(${chunk.chunk_index}, 'a')">A is better</button>
                <button class="action-btn" onclick="saveEvaluationPreference(${chunk.chunk_index}, 'b')">B is better</button>
                <button class="action-btn" onclick="saveEvaluationPreference(${chunk.chunk_index}, 'equal')">Equal</button>
                <button class="action-btn" onclick="saveEvaluationPreference(${chunk.chunk_index}, 'both_need_edit')">Both need work</button>
            </div>
            <label class="evaluation-edit field">
                <span>Approved Persian correction (optional)</span>
                <textarea id="evaluation-edit-${chunk.chunk_index}" dir="rtl" rows="4"></textarea>
            </label>
            <button class="btn btn-primary btn-compact" onclick="saveEvaluationEdit(${chunk.chunk_index})">Approve correction</button>`;
        list.appendChild(item);
    });
    document.getElementById("evaluationSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function saveEvaluationPreference(chunkIndex, preference, editedTranslation = "") {
    if (!currentEvaluationId) return;
    const response = await fetch(authUrl(
        `/api/evaluations/${currentEvaluationId}/chunks/${chunkIndex}/preference`
    ), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            preference: preference,
            edited_translation: editedTranslation,
            save_to_benchmark: preference === "a" || preference === "b" || Boolean(editedTranslation)
        })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        alert(data.error || "Could not save the comparison decision.");
        return;
    }
    await openEvaluation(currentEvaluationId);
}

async function saveEvaluationEdit(chunkIndex) {
    const field = document.getElementById(`evaluation-edit-${chunkIndex}`);
    const value = (field.value || "").trim();
    if (!value) {
        alert("Enter an approved Persian correction first.");
        return;
    }
    await saveEvaluationPreference(chunkIndex, "both_need_edit", value);
}

function downloadEvaluation(format) {
    if (!currentEvaluationId) return;
    window.open(authUrl(`/api/evaluations/${currentEvaluationId}/report?format=${format}`), "_blank");
}

async function openReview(jobId) {
    currentReviewJobId = jobId;
    fetchResearchSuggestions(jobId);
    const section = document.getElementById("reviewSection");
    const list = document.getElementById("reviewList");
    document.getElementById("reviewJobTitle").innerText = `Job ${jobId}`;
    section.hidden = false;
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
                    <span>Blocking: ${(chunk.blocking_critique_issues || []).length}</span>
                    <span>Review: ${chunk.needs_review ? "needed" : "clear"}</span>
                    <span>QA: ${chunk.qa_unavailable ? "unavailable" : "available"}</span>
                    <span>Rejected edits: ${chunk.integrity_rejections || 0}</span>
                    <span>Final integrity: ${chunk.integrity_final_failures ? "review" : "clear"}</span>
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

async function fetchResearchSuggestions(jobId) {
    const block = document.getElementById("researchBlock");
    const title = document.getElementById("researchSuggestionTitle");
    const list = document.getElementById("researchSuggestionList");
    if (!block || !title || !list) return;
    const response = await fetch(authUrl(
        "/api/jobs/" + jobId + "/research"
    ));
    if (!response.ok) return;
    const data = await response.json();
    const research = data.research;
    const terms = research && Array.isArray(research.terms)
        ? research.terms : [];
    const suggested = terms
        .map((term, index) => ({ term: term, index: index }))
        .filter(item => item.term.status === "suggested");
    if (!research) {
        block.hidden = true;
        list.innerHTML = "";
        return;
    }
    block.hidden = false;
    title.innerText = "Research suggestions - Job " + jobId;
    list.innerHTML = "";
    const metadata = document.createElement("div");
    metadata.className = "research-meta";
    const providers = Array.isArray(research.providers_used)
        ? research.providers_used.join(", ") : "";
    metadata.innerHTML =
        "<span>Status: <strong>" + escapeHtml(research.status || "unknown") +
        "</strong></span><span>Sources: <strong>" +
        escapeHtml((research.sources || []).length) +
        "</strong></span><span>Queries: <strong>" +
        escapeHtml((research.queries || []).length) +
        "</strong></span><span>Providers: <strong>" +
        escapeHtml(providers || "none") + "</strong></span>";
    list.appendChild(metadata);
    if (research.book_context) {
        const contextRow = document.createElement("div");
        contextRow.className = "research-context";
        contextRow.innerHTML = "<strong>Book context</strong><p>" +
            escapeHtml(research.book_context) + "</p>";
        list.appendChild(contextRow);
    }
    if (research.error) {
        const errorRow = document.createElement("p");
        errorRow.className = "empty-state error";
        errorRow.textContent = research.error;
        list.appendChild(errorRow);
    }
    suggested.forEach(item => {
        const row = document.createElement("div");
        row.className = "glossary-row auto";
        row.innerHTML =
            "<span><strong>" + escapeHtml(item.term.source) +
            "</strong> to " + escapeHtml(item.term.target) + "</span>" +
            "<span>" + escapeHtml(item.term.confidence || "low") +
            " | " + escapeHtml(item.term.reason || "") + "</span>" +
            "<span class='glossary-actions'><button class='action-btn' data-index='" + item.index +
            "' data-action='approve' onclick='reviewResearchTerm(" +
            "this.dataset.index, this.dataset.action)'>Approve</button>" +
            "<button class='action-btn danger' data-index='" + item.index +
            "' data-action='reject' onclick='reviewResearchTerm(" +
            "this.dataset.index, this.dataset.action)'>Reject</button></span>";
        list.appendChild(row);
    });
}

async function reviewResearchTerm(index, action) {
    if (!currentReviewJobId) return;
    const response = await fetch(authUrl(
        "/api/jobs/" + currentReviewJobId + "/research/terms/" +
        index + "/" + action
    ), { method: "POST" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        alert(data.error || "Research term update failed");
        return;
    }
    fetchResearchSuggestions(currentReviewJobId);
    fetchGlossaryTerms();
}

async function fetchGlossaryTerms() {
    const list = document.getElementById("glossaryList");
    const count = document.getElementById("glossaryCount");
    if (!list) return;
    try {
        const response = await fetch(authUrl("/api/glossary/terms"));
        if (!response.ok) throw new Error("Failed to load glossary");
        const data = await response.json();
        const terms = data.terms || [];
        if (count) count.textContent = `${terms.length} term${terms.length === 1 ? "" : "s"}`;
        list.innerHTML = "";
        if (!terms.length) {
            list.innerHTML = '<p class="empty-state">No glossary terms yet.</p>';
            return;
        }
        terms.slice(0, 80).forEach(term => {
            const row = document.createElement("div");
            row.className = `glossary-row ${term.is_auto ? "auto" : ""}`;
            row.innerHTML = `
                <span><strong>${escapeHtml(term.source)}</strong> to ${escapeHtml(term.target)}</span>
                <span>${escapeHtml(term.domain || "")} ${term.include_original ? "| English on first use" : ""}</span>
                <span class="glossary-actions">
                    ${term.is_auto ? `<button class="action-btn" onclick="approveGlossaryTerm(${term.index})">Approve</button>` : ""}
                    <button class="action-btn danger" onclick="deleteGlossaryTerm(${term.index})">Delete</button>
                </span>
            `;
            list.appendChild(row);
        });
    } catch (e) {
        if (count) count.textContent = "Unavailable";
        list.innerHTML = `<p class="empty-state error">${e.message}</p>`;
    }
}

async function addGlossaryTerm() {
    const payload = {
        source: document.getElementById("glossSource").value,
        target: document.getElementById("glossTarget").value,
        domain: document.getElementById("glossDomain").value,
        context: document.getElementById("glossContext").value,
        include_original: document.getElementById("glossIncludeOriginal").checked
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
    document.getElementById("glossIncludeOriginal").checked = false;
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
