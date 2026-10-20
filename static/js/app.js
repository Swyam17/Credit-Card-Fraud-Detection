/**
 * app.js
 * Client-side Controller for AegisGuard AI Fraud Detection Portal
 * Handles Real-time Risk Radar, Virtual Credit Card, Banking Simulator,
 * Chart.js Analytics, and Batch CSV Scanning.
 */

// Global State
let accountsData = [];
let selectedAccount = null;
let charts = {};

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    await loadAccounts();
    await loadAnalytics();
    setupDropZone();
    loadTransactionsHistory();
}

/* ==========================================================================
   TAB MANAGEMENT
   ========================================================================== */

function switchTab(tabId) {
    document.querySelectorAll(".nav-tab").forEach(tab => {
        tab.classList.toggle("active", tab.dataset.tab === tabId);
    });

    document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.toggle("active", pane.id === `tab-${tabId}`);
    });

    // Re-render charts if analytics tab is opened to handle canvas resizing
    if (tabId === "analytics" && charts.hourly) {
        Object.values(charts).forEach(c => c.resize());
    }
}

/* ==========================================================================
   1. LIVE FRAUD RADAR & PRESET LOADER
   ========================================================================== */

const PRESETS = {
    coffee: {
        amount: 4.50,
        hour: 8,
        category: "Dining & Restaurants",
        type: "Contactless_NFC",
        distance: 1.2,
        failed_pins: 0,
        trust: 98,
        foreign: false,
        name: "JOHN DOE",
        brand: "VISA"
    },
    grocery: {
        amount: 78.20,
        hour: 17,
        category: "Grocery",
        type: "Chip_POS",
        distance: 4.5,
        failed_pins: 0,
        trust: 95,
        foreign: false,
        name: "JANE SMITH",
        brand: "MASTERCARD"
    },
    gas: {
        amount: 45.00,
        hour: 12,
        category: "Gas & Fuel",
        type: "Chip_POS",
        distance: 8.0,
        failed_pins: 0,
        trust: 92,
        foreign: false,
        name: "ALICE WILSON",
        brand: "AMEX"
    },
    crypto: {
        amount: 4500.00,
        hour: 3,
        category: "Crypto & Trading",
        type: "Online_CNP",
        distance: 1850.0,
        failed_pins: 2,
        trust: 18,
        foreign: true,
        name: "UNKNOWN HOLDER",
        brand: "VISA"
    },
    overseas: {
        amount: 3890.00,
        hour: 2,
        category: "Luxury & Jewelry",
        type: "Online_CNP",
        distance: 5200.0,
        failed_pins: 1,
        trust: 25,
        foreign: true,
        name: "ANONYMOUS ENTITY",
        brand: "MASTERCARD"
    },
    card_test: {
        amount: 1.25,
        hour: 23,
        category: "Online Retail",
        type: "Online_CNP",
        distance: 650.0,
        failed_pins: 3,
        trust: 30,
        foreign: true,
        name: "BOTNET SCRIPT",
        brand: "VISA"
    }
};

function loadPreset(key) {
    const p = PRESETS[key];
    if (!p) return;

    document.getElementById("tx-amount").value = p.amount.toFixed(2);
    document.getElementById("tx-hour").value = p.hour;
    document.getElementById("tx-category").value = p.category;
    document.getElementById("tx-type").value = p.type;
    document.getElementById("tx-distance").value = p.distance;
    document.getElementById("tx-failed-pins").value = p.failed_pins;
    document.getElementById("tx-device-trust").value = p.trust;
    updateTrustLabel(p.trust);
    document.getElementById("tx-is-foreign").checked = p.foreign;

    // Update 3D Virtual Card
    document.getElementById("cardholder-name").innerText = p.name;
    document.getElementById("card-brand").innerText = p.brand;

    // Trigger AI Scan
    handleDetection(new Event("submit"));
}

function updateTrustLabel(val) {
    document.getElementById("trust-val").innerText = `${val}%`;
}

async function handleDetection(e) {
    if (e && e.preventDefault) e.preventDefault();

    const btn = document.getElementById("btn-scan");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> EVALUATING RISK VECTORS...`;

    const payload = {
        amount: parseFloat(document.getElementById("tx-amount").value) || 0,
        hour: parseInt(document.getElementById("tx-hour").value) || 0,
        merchant_category: document.getElementById("tx-category").value,
        distance: parseFloat(document.getElementById("tx-distance").value) || 0,
        is_foreign: document.getElementById("tx-is-foreign").checked ? 1 : 0,
        failed_pin: parseInt(document.getElementById("tx-failed-pins").value) || 0,
        device_trust: parseInt(document.getElementById("tx-device-trust").value) || 50
    };

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.success) {
            updateRiskDisplay(data.prediction);
        } else {
            alert(`Detection Error: ${data.error}`);
        }
    } catch (err) {
        console.error("Prediction API Error:", err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-shield-virus"></i> RUN AI FRAUD INSPECTION`;
    }
}

function updateRiskDisplay(pred) {
    const probVal = document.getElementById("prob-val");
    const barFill = document.getElementById("risk-bar-fill");
    const decBadge = document.getElementById("decision-badge");
    const cardBadge = document.getElementById("card-badge");
    const cardElem = document.getElementById("virtual-card");
    const reasonList = document.getElementById("reason-list");

    // Animate percentage text
    animateValue(probVal, parseFloat(probVal.innerText) || 0, pred.fraud_percentage, 400);

    // Update Bar Fill
    barFill.style.width = `${Math.max(2, Math.min(pred.fraud_percentage, 100))}%`;
    barFill.style.backgroundColor = pred.risk_color;

    // Decision Badge
    decBadge.innerText = `${pred.recommended_action} • ${pred.risk_level} RISK`;
    decBadge.className = `decision-badge ${pred.is_fraud ? "danger" : "safe"}`;

    // Virtual Card Styling
    if (pred.is_fraud) {
        cardBadge.className = "card-status-badge badge-danger";
        cardBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> FLAGGED`;
        cardElem.classList.add("card-alert");
    } else {
        cardBadge.className = "card-status-badge badge-safe";
        cardBadge.innerHTML = `<i class="fa-solid fa-shield-check"></i> SECURE`;
        cardElem.classList.remove("card-alert");
    }

    // Explainable AI Reasons
    reasonList.innerHTML = "";
    pred.risk_reasons.forEach(r => {
        const li = document.createElement("li");
        li.className = pred.is_fraud ? "reason-alert" : "reason-safe";
        li.innerText = r;
        reasonList.appendChild(li);
    });
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = (progress * (end - start) + start).toFixed(1);
        obj.innerHTML = `${current}%`;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

/* ==========================================================================
   2. BANKING PORTAL SIMULATOR
   ========================================================================== */

async function loadAccounts() {
    try {
        const res = await fetch("/api/accounts");
        const data = await res.json();
        accountsData = data.accounts || [];

        const container = document.getElementById("account-cards-container");
        container.innerHTML = "";

        accountsData.forEach((acc, idx) => {
            const card = document.createElement("div");
            card.className = `account-card ${idx === 0 ? "selected" : ""}`;
            card.onclick = () => selectAccount(acc, card);
            card.innerHTML = `
                <div class="acc-name">${acc.name}</div>
                <div class="acc-num">${acc.card_type} •••• ${acc.card_last4}</div>
                <div class="acc-bal">$${acc.balance.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                <div class="acc-city"><i class="fa-solid fa-location-dot"></i> ${acc.city}</div>
            `;
            container.appendChild(card);
        });

        if (accountsData.length > 0) {
            selectAccount(accountsData[0], container.children[0]);
        }
    } catch (err) {
        console.error("Accounts load error:", err);
    }
}

function selectAccount(acc, cardElement) {
    selectedAccount = acc;
    document.querySelectorAll(".account-card").forEach(c => c.classList.remove("selected"));
    if (cardElement) cardElement.classList.add("selected");

    document.getElementById("transfer-acc").value = `${acc.name} (${acc.account_number}) - Bal: $${acc.balance.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
}

async function handleTransfer(e) {
    e.preventDefault();

    if (!selectedAccount) {
        alert("Please select an active customer account first.");
        return;
    }

    const payload = {
        account_number: selectedAccount.account_number,
        pin: document.getElementById("transfer-pin").value,
        recipient: document.getElementById("transfer-recipient").value,
        amount: parseFloat(document.getElementById("transfer-amount").value),
        merchant_category: document.getElementById("transfer-cat").value,
        distance: document.getElementById("transfer-location").value === "international" ? 4500 : (document.getElementById("transfer-location").value === "domestic" ? 250 : 10),
        is_foreign: document.getElementById("transfer-location").value === "international" ? 1 : 0,
        hour: parseInt(document.getElementById("transfer-time").value)
    };

    try {
        const res = await fetch("/api/bank-transfer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        showSecurityModal(data);
        await loadAccounts();
        await loadTransactionsHistory();
    } catch (err) {
        console.error("Transfer error:", err);
        alert("Transfer processing failed.");
    }
}

function showSecurityModal(data) {
    const modal = document.getElementById("security-modal");
    const icon = document.getElementById("modal-icon");
    const title = document.getElementById("modal-title");
    const msg = document.getElementById("modal-msg");
    const details = document.getElementById("modal-details");

    if (data.is_fraud) {
        icon.className = "modal-icon alert";
        icon.innerHTML = `<i class="fa-solid fa-hand-holding-dollar"></i>`;
        title.innerText = "SECURITY INTERVENTION";
        title.style.color = "#f87171";
        msg.innerText = data.message;

        details.style.display = "block";
        details.innerHTML = `
            <strong>Incident Ref:</strong> ${data.transaction_id}<br>
            <strong>Risk Probability:</strong> ${data.prediction.fraud_percentage}% (${data.prediction.risk_level})<br>
            <strong>Trigger Factors:</strong><br>
            ${data.prediction.risk_reasons.map(r => `• ${r}`).join("<br>")}
        `;
    } else if (data.success) {
        icon.className = "modal-icon safe";
        icon.innerHTML = `<i class="fa-solid fa-check"></i>`;
        title.innerText = "TRANSACTION APPROVED";
        title.style.color = "#34d399";
        msg.innerText = data.message;

        details.style.display = "block";
        details.innerHTML = `
            <strong>Auth Ref:</strong> ${data.transaction_id}<br>
            <strong>Remaining Balance:</strong> $${data.remaining_balance.toLocaleString('en-US', {minimumFractionDigits: 2})}<br>
            <strong>AI Shield:</strong> Verified Legitimate (Risk: ${data.prediction.fraud_percentage}%)
        `;
    } else {
        icon.className = "modal-icon alert";
        icon.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i>`;
        title.innerText = "TRANSACTION FAILED";
        title.style.color = "#fbbf24";
        msg.innerText = data.message;
        details.style.display = "none";
    }

    modal.classList.add("active");
}

function closeModal() {
    document.getElementById("security-modal").classList.remove("active");
}

async function loadTransactionsHistory() {
    try {
        const res = await fetch("/api/transactions-history");
        const data = await res.json();
        const tbody = document.getElementById("audit-table-body");
        tbody.innerHTML = "";

        (data.history || []).forEach(tx => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><code>${tx.id}</code></td>
                <td><strong>${tx.recipient}</strong></td>
                <td>$${tx.amount.toFixed(2)}</td>
                <td><span style="color: ${tx.status === 'BLOCKED' ? '#f87171' : '#34d399'}">${tx.prob}%</span></td>
                <td><span class="badge-status ${tx.status.toLowerCase()}">${tx.status}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Ledger history error:", err);
    }
}

/* ==========================================================================
   3. MODEL & DATA ANALYTICS CHARTS
   ========================================================================== */

async function loadAnalytics() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();

        const m = data.metrics;
        document.getElementById("stat-total-records").innerText = m.dataset_total.toLocaleString();
        document.getElementById("stat-fraud-rate").innerText = `${m.fraud_percentage}%`;
        document.getElementById("stat-accuracy").innerText = `${(m.accuracy * 100).toFixed(1)}%`;
        document.getElementById("stat-roc-auc").innerText = m.roc_auc.toFixed(3);

        renderHourlyChart(data.hourly);
        renderClassChart(m);
        renderCategoriesChart(data.categories);
        renderConfusionMatrix(m.confusion_matrix);
    } catch (err) {
        console.error("Analytics fetch error:", err);
    }
}

function renderHourlyChart(hourly) {
    const ctx = document.getElementById("chart-hourly").getContext("2d");
    if (charts.hourly) charts.hourly.destroy();

    charts.hourly = new Chart(ctx, {
        type: "bar",
        data: {
            labels: hourly.hours,
            datasets: [
                {
                    label: "Fraudulent",
                    data: hourly.fraud,
                    backgroundColor: "#ef4444",
                    borderRadius: 4
                },
                {
                    label: "Normal",
                    data: hourly.normal,
                    backgroundColor: "rgba(99, 102, 241, 0.4)",
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", maxTicksLimit: 12 } },
                y: { stacked: true, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }
            },
            plugins: {
                legend: { labels: { color: "#e2e8f0" } }
            }
        }
    });
}

function renderClassChart(m) {
    const ctx = document.getElementById("chart-class").getContext("2d");
    if (charts.class) charts.class.destroy();

    charts.class = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Legitimate (96.5%)", "Fraudulent (3.5%)"],
            datasets: [{
                data: [m.dataset_normal, m.dataset_fraud],
                backgroundColor: ["#10b981", "#ef4444"],
                borderColor: "#090d16",
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom", labels: { color: "#e2e8f0" } }
            }
        }
    });
}

function renderCategoriesChart(cats) {
    const ctx = document.getElementById("chart-categories").getContext("2d");
    if (charts.cats) charts.cats.destroy();

    const labels = Object.keys(cats);
    const fraudCounts = labels.map(k => cats[k].fraud);

    charts.cats = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Fraud Count",
                data: fraudCounts,
                backgroundColor: "#a855f7",
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
                y: { grid: { display: false }, ticks: { color: "#cbd5e1" } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderConfusionMatrix(cm) {
    const container = document.getElementById("cm-display");
    if (!cm || cm.length < 2) return;

    container.innerHTML = `
        <div class="cm-grid">
            <div class="cm-cell tn">
                <div class="cm-val">${cm[0][0]}</div>
                <div class="cm-lbl">True Negative (Safe)</div>
            </div>
            <div class="cm-cell">
                <div class="cm-val">${cm[0][1]}</div>
                <div class="cm-lbl">False Positive</div>
            </div>
            <div class="cm-cell">
                <div class="cm-val">${cm[1][0]}</div>
                <div class="cm-lbl">False Negative</div>
            </div>
            <div class="cm-cell tp">
                <div class="cm-val">${cm[1][1]}</div>
                <div class="cm-lbl">True Positive (Fraud)</div>
            </div>
        </div>
    `;
}

/* ==========================================================================
   4. BATCH CSV SCANNER
   ========================================================================== */

function triggerFileInput() {
    document.getElementById("batch-file-input").click();
}

function setupDropZone() {
    const zone = document.getElementById("drop-zone");
    ["dragenter", "dragover"].forEach(name => {
        zone.addEventListener(name, (e) => {
            e.preventDefault();
            zone.style.borderColor = "#6366f1";
            zone.style.background = "rgba(99, 102, 241, 0.15)";
        });
    });

    ["dragleave", "drop"].forEach(name => {
        zone.addEventListener(name, (e) => {
            e.preventDefault();
            zone.style.borderColor = "rgba(99, 102, 241, 0.4)";
            zone.style.background = "rgba(99, 102, 241, 0.03)";
        });
    });

    zone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadBatchCSV(files[0]);
        }
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadBatchCSV(file);
    }
}

async function uploadBatchCSV(file) {
    const formData = new FormData();
    formData.append("file", file);

    const zone = document.getElementById("drop-zone");
    const origHTML = zone.innerHTML;
    zone.innerHTML = `<i class="fa-solid fa-spinner fa-spin upload-icon"></i><h3>Scanning Transactions with AI Shield...</h3>`;

    try {
        const res = await fetch("/api/batch-predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if (data.success) {
            displayBatchResults(data);
        } else {
            alert(`Batch error: ${data.message}`);
        }
    } catch (err) {
        console.error("Batch upload failed:", err);
        alert("Failed to process batch CSV.");
    } finally {
        zone.innerHTML = origHTML;
    }
}

function displayBatchResults(data) {
    document.getElementById("batch-results-wrap").style.display = "block";
    document.getElementById("batch-total").innerText = data.total_evaluated;
    document.getElementById("batch-frauds").innerText = data.fraud_detected;
    document.getElementById("batch-legit").innerText = data.legitimate_detected;
    document.getElementById("batch-blocked-money").innerText = `$${data.total_blocked_dollars.toLocaleString('en-US', {minimumFractionDigits: 2})}`;

    const tbody = document.getElementById("batch-table-body");
    tbody.innerHTML = "";

    (data.records || []).slice(0, 50).forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>#${r.row_id}</td>
            <td><strong>$${r.amount.toFixed(2)}</strong></td>
            <td>${r.hour}:00</td>
            <td>${r.merchant_category}</td>
            <td><strong style="color: ${r.is_fraud ? '#f87171' : '#34d399'}">${r.probability}%</strong></td>
            <td><span class="badge-status ${r.is_fraud ? 'blocked' : 'approved'}">${r.risk_level}</span></td>
            <td><code>${r.action}</code></td>
        `;
        tbody.appendChild(tr);
    });
}

/* ==========================================================================
   5. DATASET SYNTHESIZER
   ========================================================================== */

async function handleDatasetGenerate(e) {
    e.preventDefault();
    const btn = document.getElementById("btn-generate");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Synthesizing New Dataset...`;

    const payload = {
        samples: parseInt(document.getElementById("gen-samples").value),
        fraud_percentage: parseFloat(document.getElementById("gen-fraud-ratio").value)
    };

    try {
        const res = await fetch("/api/generate-dataset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.success) {
            alert(`Success! ${data.message}`);
            await loadAnalytics();
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (err) {
        console.error("Dataset generation error:", err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-gears"></i> GENERATE & RE-SYNTHESIZE DATASET`;
    }
}
