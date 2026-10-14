/**
 * ChatGPT vs Gemini — Dual-Model Comparison & Evaluation Suite
 * Vanilla JavaScript Controller
 * Course: Cloud Computing (IPCC22IS72) — AAT Project
 */

// =============================================================================
// Benchmark Test Prompts (10 Representative Cloud & AI Scenarios)
// =============================================================================

const BENCHMARK_PROMPTS = [
  {
    id: 1,
    category: "Cloud",
    prompt: "Explain how an API Gateway handles rate limiting, authentication, and reverse proxy routing in a cloud-native architecture.",
  },
  {
    id: 2,
    category: "Factual QA",
    prompt: "What are the primary architectural differences between object storage and block storage in cloud environments?",
  },
  {
    id: 3,
    category: "Summarization",
    prompt: "Summarize the core principles of the Twelve-Factor App methodology in three concise, actionable bullet points.",
  },
  {
    id: 4,
    category: "Code Gen",
    prompt: "Write an asynchronous Python function using httpx that performs an HTTP GET request with exponential backoff retry.",
  },
  {
    id: 5,
    category: "Reasoning",
    prompt: "Service A calls Service B with a 2-second timeout, and Service B calls Service C with a 3-second timeout. If Service C hangs for 4 seconds, trace what occurs at Service A and B. Explain how to prevent this cascading failure.",
  },
  {
    id: 6,
    category: "AI Concepts",
    prompt: "Explain the difference between temperature, top-p (nucleus sampling), and top-k in generative large language model decoding.",
  },
  {
    id: 7,
    category: "Multilingual",
    prompt: "Explain what cloud elasticity means in French, Spanish, and German in one clear sentence each.",
  },
  {
    id: 8,
    category: "Security",
    prompt: "What is the Principle of Least Privilege (PoLP) and how is it implemented using Cloud IAM role-based access control (RBAC)?",
  },
  {
    id: 9,
    category: "Data Transform",
    prompt: 'Convert this JSON user object into a clean SQL INSERT statement: {"id": 101, "name": "Alice", "role": "Admin", "active": true}',
  },
  {
    id: 10,
    category: "Cost Optimization",
    prompt: "Provide three concrete strategies to optimize cloud computing spend for serverless container workloads deployed on Google Cloud Run.",
  },
];

// Compatibility reference for /api/generate
const API_GENERATE_ENDPOINT = "/api/generate";
const API_COMPARE_ENDPOINT = "/api/compare";

// Benchmark state
let benchmarkResults = [];
let isBenchmarkRunning = false;
let sessionHistory = [];
let manualEvaluations = {};

// =============================================================================
// DOM Elements Initialization
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Navigation & Health
  const statusDot = document.getElementById("status-dot");
  const statusLabel = document.getElementById("status-label");
  const geminiDot = document.getElementById("gemini-dot");
  const geminiStatusVal = document.getElementById("gemini-status-val");
  const groqDot = document.getElementById("groq-dot");
  const groqStatusVal = document.getElementById("groq-status-val");
  const btnThemeToggle = document.getElementById("btn-theme-toggle");

  // Tabs
  const tabPlayground = document.getElementById("tab-playground");
  const tabEvaluation = document.getElementById("tab-evaluation");
  const viewPlayground = document.getElementById("view-playground");
  const viewEvaluation = document.getElementById("view-evaluation");

  // Prompt Composer
  const promptInput = document.getElementById("prompt-input");
  const charCounter = document.getElementById("char-counter");
  const btnClearPrompt = document.getElementById("btn-clear-prompt");
  const btnGenerate = document.getElementById("btn-generate");
  const generateSpinner = document.getElementById("generate-spinner");
  const generateBtnText = document.getElementById("generate-btn-text");
  const playgroundError = document.getElementById("playground-error");
  const playgroundErrorTitle = document.getElementById("playground-error-title");
  const playgroundErrorMsg = document.getElementById("playground-error-msg");
  const btnCloseError = document.getElementById("btn-close-error");
  const presetChips = document.getElementById("preset-chips");
  const modelSelect = document.getElementById("model-select");
  const groqModelSelect = document.getElementById("groq-model-select");
  const timeoutSelect = document.getElementById("timeout-select");

  // ChatGPT Card Elements
  const chatgptModelBadge = document.getElementById("chatgpt-model-badge");
  const chatgptLatency = document.getElementById("chatgpt-latency");
  const chatgptInputTokens = document.getElementById("chatgpt-input-tokens");
  const chatgptOutputTokens = document.getElementById("chatgpt-output-tokens");
  const chatgptTotalTokens = document.getElementById("chatgpt-total-tokens");
  const chatgptEmptyState = document.getElementById("chatgpt-empty-state");
  const chatgptLoadingState = document.getElementById("chatgpt-loading-state");
  const chatgptTimer = document.getElementById("chatgpt-timer");
  const chatgptErrorState = document.getElementById("chatgpt-error-state");
  const chatgptErrorMsg = document.getElementById("chatgpt-error-msg");
  const chatgptContentArea = document.getElementById("chatgpt-content-area");
  const chatgptText = document.getElementById("chatgpt-text");
  const btnCopyChatgpt = document.getElementById("btn-copy-chatgpt");

  // Gemini Card Elements
  const geminiModelBadge = document.getElementById("gemini-model-badge");
  const geminiLatency = document.getElementById("gemini-latency");
  const geminiInputTokens = document.getElementById("gemini-input-tokens");
  const geminiOutputTokens = document.getElementById("gemini-output-tokens");
  const geminiTotalTokens = document.getElementById("gemini-total-tokens");
  const geminiEmptyState = document.getElementById("gemini-empty-state");
  const geminiLoadingState = document.getElementById("gemini-loading-state");
  const geminiTimer = document.getElementById("gemini-timer");
  const geminiErrorState = document.getElementById("gemini-error-state");
  const geminiErrorMsg = document.getElementById("gemini-error-msg");
  const geminiContentArea = document.getElementById("gemini-content-area");
  const geminiText = document.getElementById("gemini-text");
  const btnCopyGemini = document.getElementById("btn-copy-gemini");

  // Summary Card
  const comparisonSummaryCard = document.getElementById("comparison-summary-card");
  const summaryWinnerBadge = document.getElementById("summary-winner-badge");
  const summaryChatgptLatency = document.getElementById("summary-chatgpt-latency");
  const summaryGeminiLatency = document.getElementById("summary-gemini-latency");
  const summaryLatencyDiff = document.getElementById("summary-latency-diff");
  const summaryLatencyPercent = document.getElementById("summary-latency-percent");
  const summaryTotalTokens = document.getElementById("summary-total-tokens");

  // Manual Evaluation
  const manualEvalCard = document.getElementById("manual-eval-card");
  const btnSaveEvaluation = document.getElementById("btn-save-evaluation");
  const evalObservation = document.getElementById("eval-observation");

  // History
  const historyList = document.getElementById("history-list");
  const btnClearHistory = document.getElementById("btn-clear-history");

  // Benchmark Tab
  const checkAll = document.getElementById("check-all");
  const btnSelectAll = document.getElementById("btn-select-all");
  const btnDeselectAll = document.getElementById("btn-deselect-all");
  const btnRunBenchmarks = document.getElementById("btn-run-benchmarks");
  const benchSpinner = document.getElementById("bench-spinner");
  const benchBtnText = document.getElementById("bench-btn-text");
  const evalTimeoutSelect = document.getElementById("eval-timeout-select");
  const btnExportCsv = document.getElementById("btn-export-csv");
  const benchmarkTableBody = document.getElementById("benchmark-table-body");
  const chartContainer = document.getElementById("chart-container");

  // Benchmark KPIs
  const kpiTotal = document.getElementById("kpi-total");
  const kpiCompleted = document.getElementById("kpi-completed");
  const kpiFailed = document.getElementById("kpi-failed");
  const kpiSuccessRate = document.getElementById("kpi-success-rate");
  const kpiChatgptMedian = document.getElementById("kpi-chatgpt-median");
  const kpiMedianLatency = document.getElementById("kpi-median-latency");
  const kpiMeanLatency = document.getElementById("kpi-mean-latency");
  const kpiTotalTokens = document.getElementById("kpi-total-tokens");

  // Modal Elements
  const inspectModal = document.getElementById("inspect-modal");
  const modalPromptText = document.getElementById("modal-prompt-text");
  const modalChatgptText = document.getElementById("modal-chatgpt-text");
  const modalGeminiText = document.getElementById("modal-gemini-text");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const btnModalDismiss = document.getElementById("btn-modal-dismiss");

  // =============================================================================
  // Health & Service Check
  // =============================================================================

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        statusDot.className = "status-dot dot-healthy";
        statusLabel.textContent = "Online";

        // Gemini Status
        if (data.gemini_configured || data.api_key_configured) {
          geminiDot.className = "meta-dot dot-ok";
          geminiStatusVal.textContent = "Configured";
        } else {
          geminiDot.className = "meta-dot dot-error";
          geminiStatusVal.textContent = "Missing Key";
        }

        // Groq Status
        if (data.groq_configured) {
          groqDot.className = "meta-dot dot-ok";
          groqStatusVal.textContent = "Configured";
        } else {
          groqDot.className = "meta-dot dot-error";
          groqStatusVal.textContent = "Missing Key";
        }

        if (data.default_model && modelSelect) {
          modelSelect.value = data.default_model;
          if (geminiModelBadge) geminiModelBadge.textContent = data.default_model;
        }
        if (data.groq_model && groqModelSelect) {
          groqModelSelect.value = data.groq_model;
          if (chatgptModelBadge) chatgptModelBadge.textContent = data.groq_model;
        }
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      statusDot.className = "status-dot dot-error";
      statusLabel.textContent = "Offline / Error";
      geminiDot.className = "meta-dot dot-unknown";
      groqDot.className = "meta-dot dot-unknown";
    }
  }

  checkHealth();

  // =============================================================================
  // Theme Toggle
  // =============================================================================

  if (btnThemeToggle) {
    const savedTheme = localStorage.getItem("ai_compare_theme") || "theme-dark";
    document.body.className = savedTheme;

    btnThemeToggle.addEventListener("click", () => {
      if (document.body.classList.contains("theme-dark")) {
        document.body.className = "theme-midnight";
        localStorage.setItem("ai_compare_theme", "theme-midnight");
      } else {
        document.body.className = "theme-dark";
        localStorage.setItem("ai_compare_theme", "theme-dark");
      }
    });
  }

  // =============================================================================
  // Tab Switcher
  // =============================================================================

  function switchTab(target) {
    if (target === "playground") {
      tabPlayground.classList.add("active");
      tabPlayground.setAttribute("aria-selected", "true");
      tabEvaluation.classList.remove("active");
      tabEvaluation.setAttribute("aria-selected", "false");
      viewPlayground.classList.add("active");
      viewEvaluation.classList.remove("active");
    } else {
      tabEvaluation.classList.add("active");
      tabEvaluation.setAttribute("aria-selected", "true");
      tabPlayground.classList.remove("active");
      tabPlayground.setAttribute("aria-selected", "false");
      viewEvaluation.classList.add("active");
      viewPlayground.classList.remove("active");
    }
  }

  tabPlayground.addEventListener("click", () => switchTab("playground"));
  tabEvaluation.addEventListener("click", () => switchTab("evaluation"));

  // =============================================================================
  // Prompt Input & Validation
  // =============================================================================

  function updateCharCount() {
    const len = promptInput.value.length;
    charCounter.textContent = `${len} / 4000 characters`;
    if (len > 4000) {
      charCounter.style.color = "var(--danger)";
    } else {
      charCounter.style.color = "var(--text-muted)";
    }
  }

  promptInput.addEventListener("input", updateCharCount);

  btnClearPrompt.addEventListener("click", () => {
    promptInput.value = "";
    updateCharCount();
    promptInput.focus();
    hidePlaygroundError();
  });

  // Preset Chips
  if (presetChips) {
    presetChips.addEventListener("click", (e) => {
      const chip = e.target.closest(".chip-btn");
      if (!chip) return;
      promptInput.value = chip.getAttribute("data-prompt") || "";
      updateCharCount();
      promptInput.focus();
      hidePlaygroundError();
    });
  }

  // Keyboard Shortcut: Ctrl+Enter or Cmd+Enter submits prompt
  promptInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      btnGenerate.click();
    }
  });

  // Error Alert Helpers
  function showPlaygroundError(title, message) {
    playgroundErrorTitle.textContent = title;
    playgroundErrorMsg.textContent = message;
    playgroundError.classList.remove("hidden");
  }

  function hidePlaygroundError() {
    playgroundError.classList.add("hidden");
  }

  btnCloseError.addEventListener("click", hidePlaygroundError);

  // Model select changes update badge labels
  modelSelect.addEventListener("change", () => {
    geminiModelBadge.textContent = modelSelect.value;
  });

  groqModelSelect.addEventListener("change", () => {
    chatgptModelBadge.textContent = groqModelSelect.value;
  });

  // =============================================================================
  // Safe Text Rendering Helper
  // =============================================================================

  function setSafeText(targetElement, text) {
    targetElement.textContent = text || "";
  }

  // =============================================================================
  // Dual Comparison Execution
  // =============================================================================

  let timerInterval = null;

  async function executeComparison(promptText, customGeminiModel = null, customGroqModel = null, customTimeout = null) {
    const prompt = (promptText || promptInput.value).trim();

    if (!prompt) {
      showPlaygroundError("Validation Error", "Prompt cannot be empty or contain only whitespace.");
      promptInput.focus();
      return null;
    }

    if (prompt.length > 4000) {
      showPlaygroundError("Validation Error", "Prompt exceeds maximum length of 4000 characters.");
      return null;
    }

    hidePlaygroundError();

    const gModel = customGeminiModel || modelSelect.value;
    const qModel = customGroqModel || groqModelSelect.value;
    const timeoutSec = customTimeout || parseFloat(timeoutSelect.value) || 30.0;

    // Enter Loading State
    btnGenerate.disabled = true;
    generateSpinner.classList.remove("hidden");
    generateBtnText.textContent = "⚡ Comparing Models...";

    // Reset ChatGPT Card UI
    chatgptEmptyState.classList.add("hidden");
    chatgptContentArea.classList.add("hidden");
    chatgptErrorState.classList.add("hidden");
    btnCopyChatgpt.classList.add("hidden");
    chatgptLoadingState.classList.remove("hidden");
    chatgptLatency.textContent = "-- ms";
    chatgptInputTokens.textContent = "--";
    chatgptOutputTokens.textContent = "--";
    chatgptTotalTokens.textContent = "--";
    chatgptModelBadge.textContent = qModel;

    // Reset Gemini Card UI
    geminiEmptyState.classList.add("hidden");
    geminiContentArea.classList.add("hidden");
    geminiErrorState.classList.add("hidden");
    btnCopyGemini.classList.add("hidden");
    geminiLoadingState.classList.remove("hidden");
    geminiLatency.textContent = "-- ms";
    geminiInputTokens.textContent = "--";
    geminiOutputTokens.textContent = "--";
    geminiTotalTokens.textContent = "--";
    geminiModelBadge.textContent = gModel;

    // Reset Summary & Manual Eval Cards
    comparisonSummaryCard.classList.add("hidden");
    manualEvalCard.classList.add("hidden");

    const startTime = performance.now();
    timerInterval = setInterval(() => {
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
      chatgptTimer.textContent = `Elapsed: ${elapsed}s`;
      geminiTimer.textContent = `Elapsed: ${elapsed}s`;
    }, 100);

    try {
      const payload = {
        prompt: prompt,
        gemini_model: gModel,
        groq_model: qModel,
        timeout_seconds: timeoutSec,
      };

      const response = await fetch(API_COMPARE_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      clearInterval(timerInterval);

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned HTTP ${response.status}`);
      }

      const data = await response.json();

      // Render ChatGPT (Groq) Result
      chatgptLoadingState.classList.add("hidden");
      if (data.chatgpt && data.chatgpt.success) {
        setSafeText(chatgptText, data.chatgpt.response_text);
        chatgptContentArea.classList.remove("hidden");
        btnCopyChatgpt.classList.remove("hidden");
        chatgptLatency.textContent = `${data.chatgpt.latency_ms} ms`;
        chatgptInputTokens.textContent = data.chatgpt.input_tokens != null ? data.chatgpt.input_tokens : "--";
        chatgptOutputTokens.textContent = data.chatgpt.output_tokens != null ? data.chatgpt.output_tokens : "--";
        chatgptTotalTokens.textContent = data.chatgpt.total_tokens != null ? data.chatgpt.total_tokens : "--";
      } else {
        chatgptErrorState.classList.remove("hidden");
        chatgptErrorMsg.textContent = (data.chatgpt && data.chatgpt.error_message) || "ChatGPT provider request failed.";
      }

      // Render Gemini Result
      geminiLoadingState.classList.add("hidden");
      if (data.gemini && data.gemini.success) {
        setSafeText(geminiText, data.gemini.response_text);
        geminiContentArea.classList.remove("hidden");
        btnCopyGemini.classList.remove("hidden");
        geminiLatency.textContent = `${data.gemini.latency_ms} ms`;
        geminiInputTokens.textContent = data.gemini.input_tokens != null ? data.gemini.input_tokens : "--";
        geminiOutputTokens.textContent = data.gemini.output_tokens != null ? data.gemini.output_tokens : "--";
        geminiTotalTokens.textContent = data.gemini.total_tokens != null ? data.gemini.total_tokens : "--";
      } else {
        geminiErrorState.classList.remove("hidden");
        geminiErrorMsg.textContent = (data.gemini && data.gemini.error_message) || "Gemini provider request failed.";
      }

      // Populate Summary Card
      if (data.chatgpt && data.gemini) {
        summaryChatgptLatency.textContent = data.chatgpt.latency_ms != null ? `${data.chatgpt.latency_ms} ms` : "Failed";
        summaryGeminiLatency.textContent = data.gemini.latency_ms != null ? `${data.gemini.latency_ms} ms` : "Failed";
        summaryLatencyDiff.textContent = data.latency_diff_ms != null ? `${data.latency_diff_ms} ms` : "--";
        summaryLatencyPercent.textContent = data.latency_diff_percent != null ? `${data.latency_diff_percent}%` : "--";
        summaryTotalTokens.textContent = data.total_tokens_consumed != null ? data.total_tokens_consumed : "--";

        if (data.faster_provider) {
          if (data.faster_provider === "ChatGPT") {
            summaryWinnerBadge.textContent = `⚡ ChatGPT was ${data.latency_diff_ms} ms faster (${data.latency_diff_percent}%)`;
            summaryWinnerBadge.style.color = "var(--accent-chatgpt)";
          } else if (data.faster_provider === "Gemini") {
            summaryWinnerBadge.textContent = `⚡ Gemini was ${data.latency_diff_ms} ms faster (${data.latency_diff_percent}%)`;
            summaryWinnerBadge.style.color = "var(--accent-gemini)";
          } else {
            summaryWinnerBadge.textContent = "⚡ Identical Latency (Tie)";
            summaryWinnerBadge.style.color = "var(--text-primary)";
          }
        } else {
          summaryWinnerBadge.textContent = "Incomplete Comparison";
          summaryWinnerBadge.style.color = "var(--warning)";
        }

        comparisonSummaryCard.classList.remove("hidden");
        manualEvalCard.classList.remove("hidden");
      }

      // Save to Session History
      saveToSessionHistory(prompt, data);

      return data;
    } catch (err) {
      clearInterval(timerInterval);
      chatgptLoadingState.classList.add("hidden");
      geminiLoadingState.classList.add("hidden");
      chatgptEmptyState.classList.remove("hidden");
      geminiEmptyState.classList.remove("hidden");
      showPlaygroundError("Comparison Request Error", err.message || "Failed to communicate with comparison gateway.");
      return null;
    } finally {
      btnGenerate.disabled = false;
      generateSpinner.classList.add("hidden");
      generateBtnText.textContent = "⚡ Compare Responses";
    }
  }

  btnGenerate.addEventListener("click", () => executeComparison());

  // =============================================================================
  // Copy to Clipboard
  // =============================================================================

  function copyTextWithFeedback(btn, textToCopy) {
    navigator.clipboard.writeText(textToCopy).then(() => {
      const orig = btn.textContent;
      btn.textContent = "✓ Copied!";
      btn.style.color = "var(--success)";
      setTimeout(() => {
        btn.textContent = orig;
        btn.style.color = "";
      }, 2000);
    });
  }

  btnCopyChatgpt.addEventListener("click", () => {
    copyTextWithFeedback(btnCopyChatgpt, chatgptText.textContent);
  });

  btnCopyGemini.addEventListener("click", () => {
    copyTextWithFeedback(btnCopyGemini, geminiText.textContent);
  });

  // =============================================================================
  // Star Rating & Manual Evaluation
  // =============================================================================

  let currentEvaluation = {
    chatgpt: { relevance: 4, clarity: 4, completeness: 4 },
    gemini: { relevance: 4, clarity: 4, completeness: 4 },
  };

  document.querySelectorAll(".star-rating").forEach((ratingGroup) => {
    ratingGroup.addEventListener("click", (e) => {
      const star = e.target.closest(".star-btn");
      if (!star) return;
      const val = parseInt(star.getAttribute("data-val"), 10);
      const provider = ratingGroup.getAttribute("data-provider");
      const metric = ratingGroup.getAttribute("data-metric");

      currentEvaluation[provider][metric] = val;

      ratingGroup.querySelectorAll(".star-btn").forEach((b) => {
        const bVal = parseInt(b.getAttribute("data-val"), 10);
        if (bVal <= val) {
          b.classList.add("active");
        } else {
          b.classList.remove("active");
        }
      });
    });
  });

  btnSaveEvaluation.addEventListener("click", () => {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    manualEvaluations[prompt] = {
      chatgpt: { ...currentEvaluation.chatgpt },
      gemini: { ...currentEvaluation.gemini },
      observation: (evalObservation.value || "").trim(),
      timestamp: new Date().toISOString(),
    };

    const orig = btnSaveEvaluation.textContent;
    btnSaveEvaluation.textContent = "✓ Saved!";
    setTimeout(() => {
      btnSaveEvaluation.textContent = orig;
    }, 2000);
  });

  // =============================================================================
  // Session History
  // =============================================================================

  function saveToSessionHistory(prompt, data) {
    const entry = {
      id: Date.now(),
      prompt: prompt,
      chatgptModel: (data.chatgpt && data.chatgpt.model) || "ChatGPT",
      geminiModel: (data.gemini && data.gemini.model) || "Gemini",
      chatgptLatency: data.chatgpt ? data.chatgpt.latency_ms : null,
      geminiLatency: data.gemini ? data.gemini.latency_ms : null,
      chatgptSuccess: data.chatgpt ? data.chatgpt.success : false,
      geminiSuccess: data.gemini ? data.gemini.success : false,
      chatgptText: (data.chatgpt && data.chatgpt.response_text) || "",
      geminiText: (data.gemini && data.gemini.response_text) || "",
      faster: data.faster_provider || "--",
      timestamp: new Date().toLocaleTimeString(),
    };

    sessionHistory.unshift(entry);
    if (sessionHistory.length > 10) sessionHistory.pop();
    renderSessionHistory();
  }

  function renderSessionHistory() {
    if (!historyList) return;
    if (sessionHistory.length === 0) {
      historyList.innerHTML = '<p class="history-empty">No comparison history recorded in this session yet.</p>';
      return;
    }

    historyList.innerHTML = sessionHistory.map((item, idx) => `
      <div class="history-item" data-idx="${idx}">
        <span class="history-prompt-text" title="${escapeHtml(item.prompt)}">${escapeHtml(item.prompt)}</span>
        <div class="history-metrics">
          <span>GPT: ${item.chatgptLatency != null ? item.chatgptLatency + 'ms' : 'Fail'}</span>
          <span>&bull;</span>
          <span>Gem: ${item.geminiLatency != null ? item.geminiLatency + 'ms' : 'Fail'}</span>
          <span>&bull;</span>
          <strong style="color: ${item.faster === 'ChatGPT' ? 'var(--accent-chatgpt)' : 'var(--accent-gemini)'}">${item.faster}</strong>
        </div>
      </div>
    `).join("");

    historyList.querySelectorAll(".history-item").forEach((el) => {
      el.addEventListener("click", () => {
        const idx = parseInt(el.getAttribute("data-idx"), 10);
        const item = sessionHistory[idx];
        if (!item) return;

        promptInput.value = item.prompt;
        updateCharCount();

        // Restore views
        if (item.chatgptSuccess) {
          setSafeText(chatgptText, item.chatgptText);
          chatgptContentArea.classList.remove("hidden");
          chatgptEmptyState.classList.add("hidden");
          btnCopyChatgpt.classList.remove("hidden");
          chatgptLatency.textContent = `${item.chatgptLatency} ms`;
        }
        if (item.geminiSuccess) {
          setSafeText(geminiText, item.geminiText);
          geminiContentArea.classList.remove("hidden");
          geminiEmptyState.classList.add("hidden");
          btnCopyGemini.classList.remove("hidden");
          geminiLatency.textContent = `${item.geminiLatency} ms`;
        }

        comparisonSummaryCard.classList.remove("hidden");
        summaryWinnerBadge.textContent = `Historical Winner: ${item.faster}`;
        summaryChatgptLatency.textContent = item.chatgptLatency != null ? `${item.chatgptLatency} ms` : "Fail";
        summaryGeminiLatency.textContent = item.geminiLatency != null ? `${item.geminiLatency} ms` : "Fail";
      });
    });
  }

  if (btnClearHistory) {
    btnClearHistory.addEventListener("click", () => {
      sessionHistory = [];
      renderSessionHistory();
    });
  }

  // =============================================================================
  // Benchmark Tab & Table Initialization
  // =============================================================================

  function initBenchmarkTable() {
    benchmarkResults = BENCHMARK_PROMPTS.map((p) => ({
      id: p.id,
      category: p.category,
      prompt: p.prompt,
      selected: true,
      status: "pending",
      chatgptStatus: "pending",
      chatgptLatency: null,
      chatgptInputTokens: null,
      chatgptOutputTokens: null,
      chatgptTotalTokens: null,
      chatgptResponse: null,
      chatgptError: null,
      geminiStatus: "pending",
      geminiLatency: null,
      geminiInputTokens: null,
      geminiOutputTokens: null,
      geminiTotalTokens: null,
      geminiResponse: null,
      geminiError: null,
      fasterProvider: null,
      latencyDiffMs: null,
      totalTokens: null,
    }));

    renderBenchmarkTable();
    updateKPIs();
  }

  function renderBenchmarkTable() {
    benchmarkTableBody.innerHTML = benchmarkResults.map((r) => `
      <tr data-id="${r.id}">
        <td class="col-select"><input type="checkbox" class="bench-check" data-id="${r.id}" ${r.selected ? "checked" : ""}></td>
        <td class="col-id">#${r.id}</td>
        <td class="col-category"><span class="badge badge-course">${escapeHtml(r.category)}</span></td>
        <td class="col-prompt" title="${escapeHtml(r.prompt)}">${escapeHtml(truncate(r.prompt, 55))}</td>
        <td class="col-status"><span class="badge-status status-${r.status}">${r.status}</span></td>
        <td class="col-latency font-mono">${r.chatgptLatency != null ? r.chatgptLatency + " ms" : (r.chatgptStatus === "failed" ? '<span style="color:var(--danger)">Failed</span>' : "—")}</td>
        <td class="col-latency font-mono">${r.geminiLatency != null ? r.geminiLatency + " ms" : (r.geminiStatus === "failed" ? '<span style="color:var(--danger)">Failed</span>' : "—")}</td>
        <td class="col-winner font-mono"><strong>${r.fasterProvider || "—"}</strong></td>
        <td class="col-tokens font-mono">${r.totalTokens != null ? r.totalTokens : "—"}</td>
        <td class="col-action">
          <button type="button" class="btn btn-ghost btn-sm btn-inspect" data-id="${r.id}" title="Inspect full responses">View</button>
        </td>
      </tr>
    `).join("");

    benchmarkTableBody.querySelectorAll(".bench-check").forEach((cb) => {
      cb.addEventListener("change", (e) => {
        const id = parseInt(e.target.getAttribute("data-id"), 10);
        const item = benchmarkResults.find((b) => b.id === id);
        if (item) item.selected = e.target.checked;
      });
    });

    benchmarkTableBody.querySelectorAll(".btn-inspect").forEach((b) => {
      b.addEventListener("click", () => {
        const id = parseInt(b.getAttribute("data-id"), 10);
        const item = benchmarkResults.find((r) => r.id === id);
        if (item) showInspectModal(item);
      });
    });
  }

  if (checkAll) {
    checkAll.addEventListener("change", (e) => {
      const isChecked = e.target.checked;
      benchmarkResults.forEach((r) => (r.selected = isChecked));
      renderBenchmarkTable();
    });
  }

  btnSelectAll.addEventListener("click", () => {
    benchmarkResults.forEach((r) => (r.selected = true));
    if (checkAll) checkAll.checked = true;
    renderBenchmarkTable();
  });

  btnDeselectAll.addEventListener("click", () => {
    benchmarkResults.forEach((r) => (r.selected = false));
    if (checkAll) checkAll.checked = false;
    renderBenchmarkTable();
  });

  // =============================================================================
  // Benchmark Batch Runner
  // =============================================================================

  async function runBenchmarks() {
    if (isBenchmarkRunning) return;

    const selectedTests = benchmarkResults.filter((r) => r.selected);
    if (selectedTests.length === 0) {
      alert("Please select at least one test prompt to run.");
      return;
    }

    isBenchmarkRunning = true;
    btnRunBenchmarks.disabled = true;
    benchSpinner.classList.remove("hidden");

    const timeoutSec = parseFloat(evalTimeoutSelect.value) || 30.0;
    const totalSelected = selectedTests.length;
    let completedCount = 0;

    for (const testItem of selectedTests) {
      benchBtnText.textContent = `Testing (${completedCount + 1}/${totalSelected})...`;
      testItem.status = "running";
      testItem.chatgptStatus = "running";
      testItem.geminiStatus = "running";
      renderBenchmarkTable();

      try {
        const payload = {
          prompt: testItem.prompt,
          gemini_model: modelSelect.value,
          groq_model: groqModelSelect.value,
          timeout_seconds: timeoutSec,
        };

        const res = await fetch(API_COMPARE_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          const data = await res.json();

          // ChatGPT Run
          if (data.chatgpt && data.chatgpt.success) {
            testItem.chatgptStatus = "success";
            testItem.chatgptLatency = data.chatgpt.latency_ms;
            testItem.chatgptInputTokens = data.chatgpt.input_tokens;
            testItem.chatgptOutputTokens = data.chatgpt.output_tokens;
            testItem.chatgptTotalTokens = data.chatgpt.total_tokens;
            testItem.chatgptResponse = data.chatgpt.response_text;
            testItem.chatgptError = null;
          } else {
            testItem.chatgptStatus = "failed";
            testItem.chatgptLatency = null;
            testItem.chatgptError = (data.chatgpt && data.chatgpt.error_message) || "Failed";
          }

          // Gemini Run
          if (data.gemini && data.gemini.success) {
            testItem.geminiStatus = "success";
            testItem.geminiLatency = data.gemini.latency_ms;
            testItem.geminiInputTokens = data.gemini.input_tokens;
            testItem.geminiOutputTokens = data.gemini.output_tokens;
            testItem.geminiTotalTokens = data.gemini.total_tokens;
            testItem.geminiResponse = data.gemini.response_text;
            testItem.geminiError = null;
          } else {
            testItem.geminiStatus = "failed";
            testItem.geminiLatency = null;
            testItem.geminiError = (data.gemini && data.gemini.error_message) || "Failed";
          }

          testItem.fasterProvider = data.faster_provider;
          testItem.latencyDiffMs = data.latency_diff_ms;
          testItem.totalTokens = data.total_tokens_consumed;

          if (testItem.chatgptStatus === "success" && testItem.geminiStatus === "success") {
            testItem.status = "success";
          } else if (testItem.chatgptStatus === "failed" && testItem.geminiStatus === "failed") {
            testItem.status = "failed";
          } else {
            testItem.status = "partial";
          }
        } else {
          testItem.status = "failed";
          testItem.chatgptStatus = "failed";
          testItem.geminiStatus = "failed";
          testItem.chatgptError = `HTTP ${res.status}`;
          testItem.geminiError = `HTTP ${res.status}`;
        }
      } catch (err) {
        testItem.status = "failed";
        testItem.chatgptStatus = "failed";
        testItem.geminiStatus = "failed";
        testItem.chatgptError = err.message || "Network error";
        testItem.geminiError = err.message || "Network error";
      }

      completedCount++;
      renderBenchmarkTable();
      updateKPIs();
      renderLatencyChart();

      // Brief pause between requests to prevent free-tier throttling
      await new Promise((r) => setTimeout(r, 600));
    }

    isBenchmarkRunning = false;
    btnRunBenchmarks.disabled = false;
    benchSpinner.classList.add("hidden");
    benchBtnText.textContent = "▶ Run Selected Tests";
  }

  btnRunBenchmarks.addEventListener("click", runBenchmarks);

  // =============================================================================
  // Benchmark KPI Computations
  // =============================================================================

  function updateKPIs() {
    const executed = benchmarkResults.filter((r) => r.status !== "pending" && r.status !== "running");
    kpiTotal.textContent = benchmarkResults.length;
    kpiCompleted.textContent = executed.filter((r) => r.status === "success" || r.status === "partial").length;
    kpiFailed.textContent = executed.filter((r) => r.status === "failed").length;

    if (executed.length === 0) {
      kpiSuccessRate.textContent = "-- %";
      kpiChatgptMedian.textContent = "-- ms";
      kpiMedianLatency.textContent = "-- ms";
      kpiMeanLatency.textContent = "-- ms";
      kpiTotalTokens.textContent = "--";
      return;
    }

    const successCount = executed.filter((r) => r.status === "success").length;
    const rate = ((successCount / executed.length) * 100).toFixed(1);
    kpiSuccessRate.textContent = `${rate}%`;

    // Latencies
    const chatgptLats = executed.map((r) => r.chatgptLatency).filter((l) => l != null);
    const geminiLats = executed.map((r) => r.geminiLatency).filter((l) => l != null);
    const allLats = [...chatgptLats, ...geminiLats];

    if (chatgptLats.length > 0) {
      kpiChatgptMedian.textContent = `${computeMedian(chatgptLats)} ms`;
    } else {
      kpiChatgptMedian.textContent = "-- ms";
    }

    if (geminiLats.length > 0) {
      kpiMedianLatency.textContent = `${computeMedian(geminiLats)} ms`;
    } else {
      kpiMedianLatency.textContent = "-- ms";
    }

    if (allLats.length > 0) {
      const sum = allLats.reduce((a, b) => a + b, 0);
      kpiMeanLatency.textContent = `${(sum / allLats.length).toFixed(1)} ms`;
    } else {
      kpiMeanLatency.textContent = "-- ms";
    }

    // Tokens
    const totalToks = executed.reduce((sum, r) => sum + (r.totalTokens || 0), 0);
    kpiTotalTokens.textContent = totalToks > 0 ? totalToks.toLocaleString() : "--";
  }

  function computeMedian(arr) {
    if (!arr || arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      return Math.round((sorted[mid - 1] + sorted[mid]) / 2);
    }
    return sorted[mid];
  }

  // =============================================================================
  // Visual Latency Chart Rendering
  // =============================================================================

  function renderLatencyChart() {
    if (!chartContainer) return;
    const evaluated = benchmarkResults.filter((r) => r.chatgptLatency != null || r.geminiLatency != null);

    if (evaluated.length === 0) {
      chartContainer.innerHTML = '<p class="chart-placeholder">Run benchmarks to render live side-by-side latency bars.</p>';
      return;
    }

    let maxLat = 1;
    evaluated.forEach((r) => {
      if (r.chatgptLatency && r.chatgptLatency > maxLat) maxLat = r.chatgptLatency;
      if (r.geminiLatency && r.geminiLatency > maxLat) maxLat = r.geminiLatency;
    });

    chartContainer.innerHTML = evaluated.map((r) => {
      const gptPct = r.chatgptLatency ? ((r.chatgptLatency / maxLat) * 100).toFixed(1) : 0;
      const gemPct = r.geminiLatency ? ((r.geminiLatency / maxLat) * 100).toFixed(1) : 0;

      return `
        <div class="chart-row">
          <span class="chart-row-label">#${r.id} (${r.category})</span>
          <div class="chart-bars-group">
            <div class="chart-bar-wrap" title="ChatGPT: ${r.chatgptLatency || 0} ms">
              <div class="chart-bar-fill chart-bar-chatgpt" style="width: ${gptPct}%"></div>
            </div>
            <div class="chart-bar-wrap" title="Gemini: ${r.geminiLatency || 0} ms">
              <div class="chart-bar-fill chart-bar-gemini" style="width: ${gemPct}%"></div>
            </div>
          </div>
          <span class="chart-row-val">GPT: ${r.chatgptLatency || '—'} / Gem: ${r.geminiLatency || '—'}</span>
        </div>
      `;
    }).join("");
  }

  // =============================================================================
  // Inspect Modal
  // =============================================================================

  function showInspectModal(item) {
    modalPromptText.textContent = item.prompt;
    modalChatgptText.textContent = item.chatgptResponse || (item.chatgptError ? `Error: ${item.chatgptError}` : "No response generated.");
    modalGeminiText.textContent = item.geminiResponse || (item.geminiError ? `Error: ${item.geminiError}` : "No response generated.");
    inspectModal.classList.remove("hidden");
  }

  function hideInspectModal() {
    inspectModal.classList.add("hidden");
  }

  btnCloseModal.addEventListener("click", hideInspectModal);
  btnModalDismiss.addEventListener("click", hideInspectModal);
  inspectModal.addEventListener("click", (e) => {
    if (e.target === inspectModal) hideInspectModal();
  });

  // =============================================================================
  // CSV Export
  // =============================================================================

  btnExportCsv.addEventListener("click", () => {
    const headers = [
      "ID",
      "Category",
      "Prompt",
      "ChatGPT_Status",
      "ChatGPT_Latency_ms",
      "ChatGPT_Input_Tokens",
      "ChatGPT_Output_Tokens",
      "ChatGPT_Total_Tokens",
      "Gemini_Status",
      "Gemini_Latency_ms",
      "Gemini_Input_Tokens",
      "Gemini_Output_Tokens",
      "Gemini_Total_Tokens",
      "Faster_Provider",
      "Latency_Diff_ms",
      "Manual_ChatGPT_Relevance",
      "Manual_ChatGPT_Clarity",
      "Manual_ChatGPT_Completeness",
      "Manual_Gemini_Relevance",
      "Manual_Gemini_Clarity",
      "Manual_Gemini_Completeness",
      "Reviewer_Notes",
      "ChatGPT_Error",
      "Gemini_Error",
    ];

    const rows = benchmarkResults.map((r) => {
      const evalData = manualEvaluations[r.prompt] || {};
      const cEval = evalData.chatgpt || {};
      const gEval = evalData.gemini || {};

      return [
        r.id,
        csvEscape(r.category),
        csvEscape(r.prompt),
        r.chatgptStatus,
        r.chatgptLatency != null ? r.chatgptLatency : "",
        r.chatgptInputTokens != null ? r.chatgptInputTokens : "",
        r.chatgptOutputTokens != null ? r.chatgptOutputTokens : "",
        r.chatgptTotalTokens != null ? r.chatgptTotalTokens : "",
        r.geminiStatus,
        r.geminiLatency != null ? r.geminiLatency : "",
        r.geminiInputTokens != null ? r.geminiInputTokens : "",
        r.geminiOutputTokens != null ? r.geminiOutputTokens : "",
        r.geminiTotalTokens != null ? r.geminiTotalTokens : "",
        r.fasterProvider || "",
        r.latencyDiffMs != null ? r.latencyDiffMs : "",
        cEval.relevance != null ? cEval.relevance : "",
        cEval.clarity != null ? cEval.clarity : "",
        cEval.completeness != null ? cEval.completeness : "",
        gEval.relevance != null ? gEval.relevance : "",
        gEval.clarity != null ? gEval.clarity : "",
        gEval.completeness != null ? gEval.completeness : "",
        csvEscape(evalData.observation || ""),
        csvEscape(r.chatgptError || ""),
        csvEscape(r.geminiError || ""),
      ].join(",");
    });

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `chatgpt_vs_gemini_benchmark_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  function csvEscape(str) {
    if (str == null) return '""';
    const s = String(str).replace(/"/g, '""');
    return `"${s}"`;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function truncate(str, max) {
    if (!str) return "";
    return str.length > max ? str.substring(0, max) + "..." : str;
  }

  // Initialize
  initBenchmarkTable();
});
