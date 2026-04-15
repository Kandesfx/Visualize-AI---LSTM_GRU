/**
 * GRU Demo V2.2 — Ported from LSTM V2.2
 * Cell Chain layout, Fly Vector animation, Battery Bar, Enhanced Output
 * GRU: 2 gates (Reset, Update), 5 animation steps per cell
 */

// ==================== GLOBAL STATE ====================
let demoData = null;
let currentCell = 0;
let currentSubStep = 0;
let totalCells = 0;
let detailLevel = 'basic';
let anim = new AnimationEngine();
let tooltipData = {};
let floatingPanel = null;
let cellChainBuilt = false;
let isAnimating = false;
const TOTAL_SUB_STEPS = 5;

// ==================== SCOPED ELEMENT QUERY ====================
function $(id) {
    const container = document.getElementById('active-cell-svg');
    if (container) return container.querySelector('#' + id);
    return document.getElementById(id);
}

// ==================== INIT ====================
document.addEventListener('DOMContentLoaded', () => {
    floatingPanel = new FloatingPanel('floating-panel');
    document.addEventListener('keydown', handleKeyboard);
});

function fillExample(el) {
    document.getElementById('input-text').value = el.textContent;
}

function resetDemo() {
    demoData = null;
    currentCell = 0;
    currentSubStep = 0;
    totalCells = 0;
    cellChainBuilt = false;
    isAnimating = false;
    anim.stopAutoPlay();

    document.getElementById('input-section').classList.remove('collapsed');
    document.getElementById('demo-content').classList.remove('active');
    document.getElementById('demo-content').style.display = 'none';
    document.getElementById('output-section').classList.remove('active');
    document.getElementById('input-text').value = '';
    document.getElementById('cell-section').style.display = 'none';

    floatingPanel.hide();
    document.getElementById('btn-autoplay').textContent = '▶▶ Tự Động';
}

// ==================== KEYBOARD SHORTCUTS ====================
function handleKeyboard(e) {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    switch (e.key) {
        case 'ArrowRight': case ' ':
            e.preventDefault(); if (demoData && !isAnimating) nextStep(); break;
        case 'ArrowLeft':
            e.preventDefault(); if (demoData && !isAnimating) prevStep(); break;
        case 'a': case 'A': if (demoData) toggleAutoPlay(); break;
        case '1': setDetailByKey('basic'); break;
        case '2': setDetailByKey('normal'); break;
        case '3': setDetailByKey('advanced'); break;
        case 'r': case 'R': resetDemo(); break;
        case 'Escape':
            if (document.getElementById('shortcuts-overlay').classList.contains('active'))
                toggleShortcuts();
            else window.location.href = '/';
            break;
        case '?': toggleShortcuts(); break;
        case '+': case '=': adjustSpeed(0.5); break;
        case '-': adjustSpeed(-0.5); break;
    }
}
function setDetailByKey(level) {
    detailLevel = level;
    document.querySelectorAll('.detail-btn').forEach((b, i) => {
        b.classList.toggle('active', ['basic', 'normal', 'advanced'][i] === level);
    });
}
function adjustSpeed(delta) {
    const slider = document.getElementById('speed-slider');
    let val = parseFloat(slider.value) + delta;
    val = Math.max(0.5, Math.min(3, val));
    slider.value = val; changeSpeed(val);
}
function toggleShortcuts() { document.getElementById('shortcuts-overlay').classList.toggle('active'); }
function togglePanel() { floatingPanel.toggle(); }

// ==================== START ANALYSIS ====================
async function startAnalysis() {
    const text = document.getElementById('input-text').value.trim();
    if (!text) { document.getElementById('input-text').focus(); return; }
    document.getElementById('loading-overlay').classList.add('active');
    try {
        demoData = await API.predictGRU(text);
        totalCells = demoData.tokens.length;
        document.getElementById('input-section').classList.add('collapsed');
        await Utils.sleep(400);
        const content = document.getElementById('demo-content');
        content.style.display = 'flex';
        content.classList.add('active');
        document.getElementById('loading-overlay').classList.remove('active');
        await runPhaseTokensAndEmbeddings();
        await Utils.sleep(600);
        await showCellSection();
    } catch (err) {
        document.getElementById('loading-overlay').classList.remove('active');
        alert('Lỗi: ' + err.message + '\n\nHãy đảm bảo server backend đang chạy.');
    }
}

// ==================== TOKENS & EMBEDDINGS ====================
async function runPhaseTokensAndEmbeddings() {
    const row = document.getElementById('tokens-embed-row');
    row.innerHTML = '';
    demoData.tokens.forEach((token, i) => {
        const emb = demoData.embeddings[i];
        const card = Utils.htmlEl('div', 'token-embed-card');
        card.id = `te-card-${i}`;
        card.innerHTML = `
            <span class="te-index">T${i + 1}</span>
            <span class="te-word" id="te-word-${i}">${token}</span>
            <div class="te-arrow" id="te-arrow-${i}"></div>
            <div class="te-vector" id="te-vector-${i}" onclick="showEmbedModal(${i})" title="Click xem chi tiết">
                ${Utils.summarizeVector(emb.summary)}
            </div>
            <div class="te-dim" id="te-dim-${i}">${emb.dim}d</div>
        `;
        row.appendChild(card);
    });
    const cards = row.querySelectorAll('.token-embed-card');
    for (let i = 0; i < cards.length; i++) {
        await Utils.sleep(120 / anim.speed);
        cards[i].classList.add('visible');
    }
    await Utils.sleep(300);
    for (let i = 0; i < cards.length; i++) {
        await Utils.sleep(180 / anim.speed);
        document.getElementById(`te-arrow-${i}`).classList.add('visible');
        document.getElementById(`te-vector-${i}`).classList.add('visible');
        document.getElementById(`te-dim-${i}`).classList.add('visible');
    }
}

// ==================== EMBEDDING MODAL ====================
function showEmbedModal(idx) {
    const emb = demoData.embeddings[idx];
    document.getElementById('modal-token-title').innerHTML =
        `📐 Embedding — "<span style="color:var(--accent-primary)">${emb.token}</span>"`;
    document.getElementById('modal-stats').innerHTML =
        `Chiều: <b>${emb.dim}</b> &nbsp;|&nbsp; Mean: <b>${Utils.fmt(emb.mean)}</b> &nbsp;|&nbsp; Min: <b>${Utils.fmt(emb.min)}</b> &nbsp;|&nbsp; Max: <b>${Utils.fmt(emb.max)}</b>`;
    const heatmap = document.getElementById('modal-heatmap');
    heatmap.innerHTML = '';
    const absMax = Math.max(Math.abs(emb.min), Math.abs(emb.max)) || 1;
    emb.vector.forEach(v => {
        const cell = Utils.htmlEl('div');
        const norm = (v + absMax) / (2 * absMax);
        const hue = norm > 0.5 ? 200 : 15;
        const sat = Math.abs(norm - 0.5) * 200;
        cell.style.cssText = `width:8px;height:8px;border-radius:1px;background:hsl(${hue},${sat}%,50%);`;
        cell.title = Utils.fmt(v, 4);
        heatmap.appendChild(cell);
    });
    document.getElementById('modal-values').textContent =
        '[' + emb.vector.map(v => Utils.fmt(v, 4)).join(', ') + ']';
    document.getElementById('embed-modal').classList.add('active');
}
function closeEmbedModal(e) {
    if (!e || e.target.id === 'embed-modal' || e.target.classList.contains('modal-close'))
        document.getElementById('embed-modal').classList.remove('active');
}

// ==================== CELL SECTION ====================
async function showCellSection() {
    const section = document.getElementById('cell-section');
    section.style.display = '';
    section.style.opacity = '0';
    setTimeout(() => { section.style.transition = 'opacity 0.5s'; section.style.opacity = '1'; }, 50);

    floatingPanel.show();
    buildMiniCells();
    cellChainBuilt = true;

    currentCell = 0;
    currentSubStep = 0;
    renderCellView();
    updateControls();
    Utils.scrollTo(section);

    await Utils.sleep(450);
    await flyTokenToCell(currentCell);
}

// ==================== RENDER CELL VIEW (center + sides) ====================
function renderCellView() {
    updateSentenceContext();
    updateTokenHighlight();
    updateMiniCells();
    renderMiniPreviews();
    renderActiveCellSVG();
    updateConnectors();
}

// ==================== MINI PREVIEWS ON SIDES ====================
function renderMiniPreviews() {
    const leftContainer = document.getElementById('chain-left');
    const rightContainer = document.getElementById('chain-right');

    leftContainer.innerHTML = '';
    const leftStart = Math.max(0, currentCell - 2);
    for (let i = leftStart; i < currentCell; i++) {
        const mini = createMiniPreview(i, true);
        leftContainer.appendChild(mini);
    }

    rightContainer.innerHTML = '';
    const rightEnd = Math.min(totalCells, currentCell + 5);
    for (let i = currentCell + 1; i < rightEnd; i++) {
        const mini = createMiniPreview(i, false);
        rightContainer.appendChild(mini);
    }
}

function createMiniPreview(idx, isDone) {
    const mini = Utils.htmlEl('div', `mini-preview${isDone ? ' done' : ''}`);
    mini.onclick = () => jumpToCell(idx);
    const token = demoData.tokens[idx] || '';
    mini.innerHTML = `
        <div class="mini-preview-title">Cell t=${idx + 1}</div>
        <div class="mini-preview-token">"${token}"</div>
        <div class="mini-preview-placeholder"></div>
    `;
    return mini;
}

// ==================== CONNECTORS ====================
function updateConnectors() {
    const connLeft = document.getElementById('chain-conn-left');
    const connRight = document.getElementById('chain-conn-right');
    const connLeftVals = document.getElementById('conn-left-vals');
    const connRightVals = document.getElementById('conn-right-vals');

    // GRU only has hidden state (no cell state)
    if (currentCell > 0) {
        connLeft.style.display = '';
        const prevState = demoData.cell_states[currentCell - 1];
        if (prevState) {
            connLeftVals.innerHTML = `<div>h: ${Utils.fmt(prevState.hidden_state.mean, 3)}</div>`;
            connLeftVals.classList.add('has-value');
        }
    } else {
        connLeft.style.display = 'none';
    }

    if (currentCell < totalCells - 1) {
        connRight.style.display = '';
        connRightVals.innerHTML = '<div>h: —</div>';
        connRightVals.classList.remove('has-value');
    } else {
        connRight.style.display = '';
        const currState = demoData.cell_states[currentCell];
        if (currState) {
            connRightVals.innerHTML = `<div>h(T): ${Utils.fmt(currState.hidden_state.mean, 3)}</div>`;
            connRightVals.classList.add('has-value');
        } else {
            connRightVals.innerHTML = '<div>h(T): —</div>';
            connRightVals.classList.remove('has-value');
        }
    }
}

// ==================== SENTENCE CONTEXT ====================
function updateSentenceContext() {
    const ctx = document.getElementById('sentence-context');
    if (!demoData) return;
    ctx.innerHTML = demoData.tokens.map((t, i) => {
        let cls = 'ctx-word';
        if (i < currentCell) cls += ' ctx-processed';
        else if (i === currentCell) cls += ' ctx-active';
        else cls += ' ctx-pending';
        return `<span class="${cls}">${t}</span>`;
    }).join('');
}

// ==================== TOKEN HIGHLIGHT ====================
function updateTokenHighlight() {
    document.querySelectorAll('.token-embed-card').forEach((card, i) => {
        card.classList.remove('active', 'processed');
        if (i < currentCell) card.classList.add('processed');
        else if (i === currentCell) card.classList.add('active');
    });
}

// ==================== MINI CELL BAR ====================
function buildMiniCells() {
    const container = document.getElementById('mini-cells');
    container.innerHTML = '';
    for (let i = 0; i < totalCells; i++) {
        const cell = Utils.htmlEl('div', 'mini-cell pending');
        cell.id = `mini-cell-${i}`;
        cell.textContent = i + 1;
        cell.onclick = () => jumpToCell(i);
        container.appendChild(cell);
    }
    updateMiniCells();
}

function updateMiniCells() {
    for (let i = 0; i < totalCells; i++) {
        const el = document.getElementById(`mini-cell-${i}`);
        if (!el) continue;
        el.className = 'mini-cell';
        if (i < currentCell) el.classList.add('done');
        else if (i === currentCell) el.classList.add('active');
        else el.classList.add('pending');
    }
}

function jumpToCell(idx) {
    if (idx >= totalCells || isAnimating) return;
    currentCell = idx;
    currentSubStep = 0;
    renderCellView();
    updateControls();
}

// ==================== CONTROLS ====================
function updateControls() {
    document.getElementById('step-display').textContent =
        `Cell ${currentCell + 1}/${totalCells} · ${currentSubStep}/${TOTAL_SUB_STEPS}`;
    document.getElementById('btn-prev').disabled = (currentCell === 0 && currentSubStep === 0);
    document.getElementById('btn-next').disabled = (currentCell >= totalCells);
}

function changeSpeed(val) {
    anim.setSpeed(parseFloat(val));
    document.getElementById('speed-value').textContent = val + 'x';
}

function setDetail(level, btn) {
    detailLevel = level;
    document.querySelectorAll('.detail-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

async function nextStep() {
    if (currentCell >= totalCells || isAnimating) return;
    isAnimating = true;

    if (currentSubStep < TOTAL_SUB_STEPS) {
        currentSubStep++;
        await animateSubStep(currentSubStep);
    }

    if (currentSubStep >= TOTAL_SUB_STEPS) {
        // ★ Delay to let user review completed cell output
        await Utils.sleep(2000 / anim.speed);

        currentCell++;
        currentSubStep = 0;
        if (currentCell < totalCells) {
            const activeEl = document.getElementById('chain-active');
            activeEl.style.opacity = '0';
            await Utils.sleep(300);
            renderCellView();
            activeEl.style.opacity = '1';

            // ★ Fly vector from tokens row down to cell
            await flyTokenToCell(currentCell);
        } else {
            showOutput();
        }
    }

    updateMiniCells();
    updateControls();
    isAnimating = false;
}

function prevStep() {
    if (isAnimating) return;
    if (currentSubStep > 0) {
        currentSubStep--;
    } else if (currentCell > 0) {
        currentCell--;
        currentSubStep = TOTAL_SUB_STEPS - 1;
    } else {
        return;
    }
    renderCellView();
    for (let s = 1; s <= currentSubStep; s++) {
        applySubStepInstant(s);
    }
    updateControls();
}

function toggleAutoPlay() {
    const btn = document.getElementById('btn-autoplay');
    if (anim.isPlaying) {
        anim.stopAutoPlay();
        btn.textContent = '▶▶ Tự Động';
    } else {
        btn.textContent = '⏸ Dừng';
        anim.startAutoPlay(async () => {
            await nextStep();
            await Utils.sleep(anim.getPauseDuration());
        });
    }
}

// ==================== FLY VECTOR TO CELL ====================
async function flyTokenToCell(idx) {
    const vecEl = document.getElementById(`te-vector-${idx}`);
    const cellContainer = document.getElementById('active-cell-svg');
    if (!vecEl || !cellContainer) return;

    const flyVec = document.getElementById('flying-vector');
    const flyVecText = document.getElementById('flying-vec-text');

    // ★ STEP 1: Scroll UP to tokens row, highlight active card
    const tokensSection = document.getElementById('tokens-section');
    tokensSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    await Utils.sleep(800);

    const card = document.getElementById(`te-card-${idx}`);
    if (card) { card.style.transition = 'transform 0.3s ease'; card.style.transform = 'scale(1.15)'; }
    await Utils.sleep(600);
    if (card) card.style.transform = 'scale(1)';
    await Utils.sleep(300);

    // ★ STEP 2: Show flying vector at source position
    const vecRect = vecEl.getBoundingClientRect();
    const startX = vecRect.left;
    const startY = vecRect.top + window.scrollY;

    flyVecText.textContent = Utils.summarizeVector(demoData.embeddings[idx].summary);
    flyVec.style.display = '';
    flyVec.style.position = 'absolute';
    flyVec.style.left = startX + 'px';
    flyVec.style.top = startY + 'px';
    flyVec.style.transition = 'none';
    flyVec.style.opacity = '1';
    flyVec.style.transform = '';
    flyVec.style.boxShadow = '';
    await Utils.sleep(300);

    // ★ STEP 3: Animate vector + scroll together using requestAnimationFrame
    const svg = cellContainer.querySelector('svg');
    if (svg) {
        const svgRect = svg.getBoundingClientRect();
        const scaleX = svgRect.width / 700;
        const scaleY = svgRect.height / 380;
        // x(t) is at SVG coords (350, 370) for GRU
        const endX = svgRect.left + (350 * scaleX) - 30;
        const endY = (svgRect.top + window.scrollY) + (370 * scaleY) - 10;

        const duration = 2000;

        await new Promise(resolve => {
            const t0 = performance.now();
            function easeInOutCubic(t) {
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            }
            function tick(now) {
                const elapsed = now - t0;
                const rawProgress = Math.min(elapsed / duration, 1);
                const p = easeInOutCubic(rawProgress);
                const curX = startX + (endX - startX) * p;
                const curY = startY + (endY - startY) * p;
                flyVec.style.left = curX + 'px';
                flyVec.style.top = curY + 'px';
                const targetViewportY = window.innerHeight * 0.45;
                const desiredScrollY = curY - targetViewportY;
                window.scrollTo({ top: desiredScrollY, behavior: 'auto' });
                if (rawProgress < 1) requestAnimationFrame(tick);
                else resolve();
            }
            requestAnimationFrame(tick);
        });

        // ★ Flash when vector arrives
        flyVec.style.transition = 'all 0.3s ease';
        flyVec.style.transform = 'scale(1.3)';
        flyVec.style.boxShadow = '0 0 20px var(--accent-primary)';
        await Utils.sleep(400);

        flyVec.style.transition = 'opacity 0.5s ease';
        flyVec.style.opacity = '0';
        await Utils.sleep(500);
        flyVec.style.display = 'none';
        flyVec.style.transform = '';
        flyVec.style.boxShadow = '';
        flyVec.style.position = 'fixed';
    } else {
        flyVec.style.opacity = '0';
        await Utils.sleep(300);
        flyVec.style.display = 'none';
    }
}

// ==================== RENDER ACTIVE CELL SVG ====================
function renderActiveCellSVG() {
    const state = demoData.cell_states[currentCell];
    if (!state) return;

    document.getElementById('active-cell-title').textContent = `GRU Cell t=${currentCell + 1}`;
    document.getElementById('active-cell-token').innerHTML = `"${state.token}"`;
    tooltipData = state;

    const prevState = currentCell > 0 ? demoData.cell_states[currentCell - 1] : null;
    const prevH = prevState ? Utils.fmt(prevState.hidden_state.mean, 2) : '0.00';

    const container = document.getElementById('active-cell-svg');
    const W = 700, H = 380;
    container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" class="cell-svg" xmlns="http://www.w3.org/2000/svg">
        <!-- HIDDEN STATE LANE (y=70) -->
        <text x="20" y="55" class="state-label" fill="var(--hidden-state)">h(t-1)=${prevH}</text>
        <line x1="70" y1="70" x2="630" y2="70" class="state-lane hidden-state-lane" id="hs-lane"/>
        <polygon points="630,65 645,70 630,75" fill="var(--hidden-state)" opacity="0.4"/>
        <text x="600" y="55" class="state-label" fill="var(--hidden-state)">h(t)</text>

        <!-- BATTERY BAR (Hidden State Energy for GRU) -->
        <g id="cell-battery" transform="translate(655, 8)">
            <text x="20" y="-2" class="battery-label" text-anchor="middle">Hidden</text>
            <rect x="10" y="4" width="20" height="5" rx="2" fill="var(--text-muted)" opacity="0.4"/>
            <rect x="2" y="9" width="36" height="100" rx="4" class="battery-outer"/>
            <rect x="5" y="12" width="30" height="94" id="battery-fill" class="battery-fill" rx="2"/>
            <text x="20" y="65" class="battery-pct" id="battery-pct" text-anchor="middle">0%</text>
            <text x="20" y="124" class="battery-value" id="battery-value" text-anchor="middle">0.00</text>
            <text x="20" y="140" class="battery-indicator" id="battery-indicator" text-anchor="middle"></text>
        </g>

        <!-- h(t-1) input -->
        <path d="M20,70 L70,70" class="pipe" id="pipe-ht-in"/>

        <!-- x(t) input from bottom — routes cleanly via right angles -->
        <path d="M350,370 L350,330 L100,330 L100,280" class="pipe" id="pipe-input-x"/>
        <text x="360" y="370" class="cell-label" fill="var(--accent-primary)" font-size="12" font-weight="600">x(t)</text>

        <!-- Concat point -->
        <circle cx="100" cy="280" r="10" fill="var(--bg-tertiary)" stroke="var(--border-medium)" stroke-width="2" id="concat-point"/>
        <text x="100" y="284" class="operator-text" font-size="10" fill="var(--text-secondary)">[,]</text>

        <!-- To gates — all from concat -->
        <path d="M110,280 L160,280" class="pipe" id="pipe-to-reset"/>
        <path d="M110,280 L340,280" class="pipe" id="pipe-to-update"/>
        <!-- x(t) branch to candidate: clean right-angle route, not diagonal -->
        <path d="M110,280 L110,320 L500,320 L500,285" class="pipe" id="pipe-to-cand"/>

        <!-- Reset gate up to reset-mul -->
        <path d="M200,240 L200,165" class="pipe" id="pipe-reset-up"/>

        <!-- Update gate up to operators -->
        <path d="M380,240 L380,130 L310,130 L310,85" class="pipe" id="pipe-update-to-1mz"/>
        <path d="M380,240 L380,130 L480,130 L480,85" class="pipe" id="pipe-update-to-z"/>

        <!-- Candidate up to z-mul (from TOP of candidate, right side) -->
        <path d="M555,240 L555,130 L495,130" class="pipe" id="pipe-cand-up"/>

        <!-- OPERATORS on hidden state lane -->
        <!-- (1-z) × h(t-1) -->
        <rect x="295" y="55" width="30" height="30" rx="6" class="operator-bg" id="op-1mz-mul"/>
        <text x="310" y="74" class="operator-text">×</text>
        <text x="310" y="48" class="cell-label" fill="var(--update-gate)" font-size="11">(1-z)</text>

        <!-- + add -->
        <rect x="400" y="55" width="30" height="30" rx="6" class="operator-bg" id="op-add"/>
        <text x="415" y="74" class="operator-text">+</text>

        <!-- z × h̃ -->
        <rect x="465" y="55" width="30" height="30" rx="6" class="operator-bg" id="op-z-mul"/>
        <text x="480" y="74" class="operator-text">×</text>
        <text x="480" y="48" class="cell-label" fill="var(--update-gate)" font-size="11">z</text>

        <!-- Reset × h(t-1) operator -->
        <rect x="185" y="140" width="30" height="30" rx="6" class="operator-bg" id="op-reset-mul"/>
        <text x="200" y="159" class="operator-text">×</text>

        <!-- h(t-1) branches -->
        <path d="M70,70 L70,155 L185,155" class="pipe" id="pipe-ht-to-reset-mul"/>
        <path d="M70,70 L295,70" class="pipe" id="pipe-ht-to-1mz"/>

        <!-- Reset×h(t-1) output to candidate — enters from TOP of candidate -->
        <path d="M215,155 L525,155 L525,240" class="pipe" id="pipe-reset-to-cand"/>

        <!-- ========= GATE NODES WITH ACTIVATION BADGES ========= -->

        <!-- Reset Gate -->
        <g class="gate-node" id="gate-reset" onmouseenter="showTooltip(event,'reset')" onmouseleave="hideTooltip()">
            <rect x="160" y="240" width="80" height="55" class="gate-bg reset" id="gate-bg-reset"/>
            <text x="200" y="256" class="gate-label">Reset</text>
            <text x="200" y="270" class="gate-sublabel" fill="var(--reset-gate)">r(t)</text>
            <!-- σ activation badge -->
            <circle cx="240" cy="240" r="12" fill="rgba(243,156,18,0.2)" stroke="var(--reset-gate)" stroke-width="1.5"/>
            <text x="240" y="244" text-anchor="middle" fill="var(--reset-gate)" font-family="'JetBrains Mono', monospace" font-size="14" font-weight="700">σ</text>
            <text x="200" y="310" class="gate-value" id="val-reset"></text>
        </g>

        <!-- Update Gate -->
        <g class="gate-node" id="gate-update" onmouseenter="showTooltip(event,'update')" onmouseleave="hideTooltip()">
            <rect x="340" y="240" width="80" height="55" class="gate-bg update" id="gate-bg-update"/>
            <text x="380" y="256" class="gate-label">Update</text>
            <text x="380" y="270" class="gate-sublabel" fill="var(--update-gate)">z(t)</text>
            <!-- σ activation badge -->
            <circle cx="420" cy="240" r="12" fill="rgba(155,89,182,0.2)" stroke="var(--update-gate)" stroke-width="1.5"/>
            <text x="420" y="244" text-anchor="middle" fill="var(--update-gate)" font-family="'JetBrains Mono', monospace" font-size="14" font-weight="700">σ</text>
            <text x="380" y="310" class="gate-value" id="val-update"></text>
        </g>

        <!-- Candidate -->
        <g class="gate-node" id="gate-candidate" onmouseenter="showTooltip(event,'candidate')" onmouseleave="hideTooltip()">
            <rect x="500" y="240" width="80" height="55" class="gate-bg candidate" id="gate-bg-candidate"/>
            <text x="540" y="258" class="gate-label">Candi.</text>
            <text x="540" y="272" class="gate-sublabel" fill="var(--candidate)">h̃(t)</text>
            <!-- tanh activation badge -->
            <rect x="555" y="228" width="36" height="18" rx="9" fill="rgba(69,183,209,0.2)" stroke="var(--candidate)" stroke-width="1.5"/>
            <text x="573" y="240" text-anchor="middle" fill="var(--candidate)" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="700">tanh</text>
            <text x="540" y="310" class="gate-value" id="val-candidate"></text>
        </g>

        <!-- Annotations showing data flow -->
        <text x="110" y="312" class="annotation-text" fill="var(--text-muted)" font-size="8">[h(t-1),x(t)]</text>
        <text x="200" y="130" class="annotation-text" fill="var(--reset-gate)" font-size="8">r(t)×h(t-1)</text>

        <text x="${W-15}" y="15" class="step-indicator" id="svg-step-info">Bước ${currentSubStep}/${TOTAL_SUB_STEPS}</text>
    </svg>`;

    // Set initial battery level based on h(t-1)
    const prevHVal = prevState ? prevState.hidden_state.mean : 0;
    setTimeout(() => updateBattery(prevHVal, 'init'), 100);
}

// ==================== BATTERY HELPER ====================
function updateBattery(value, mode) {
    const bf = $('battery-fill');
    const bPct = $('battery-pct');
    const bVal = $('battery-value');
    const bInd = $('battery-indicator');
    if (!bf) return;

    // Sigmoid scaling: value=0 → 50%, positive → >50%, negative → <50%
    const sigmoid = 1 / (1 + Math.exp(-value * 10));
    const pct = Math.round(sigmoid * 100);
    const fillRatio = sigmoid;

    const fillH = Math.max(2, 94 * fillRatio);
    const fillY = 12 + (94 - fillH);

    let fillColor;
    if (pct < 30) fillColor = '#ef4444';
    else if (pct < 45) fillColor = '#f59e0b';
    else if (pct < 55) fillColor = '#eab308';
    else if (pct < 70) fillColor = '#22c55e';
    else fillColor = '#10b981';

    bf.style.transition = 'all 0.8s ease';
    bf.setAttribute('y', fillY.toFixed(0));
    bf.setAttribute('height', fillH.toFixed(0));
    bf.style.fill = fillColor;
    bf.style.opacity = '0.75';

    if (bPct) bPct.textContent = pct + '%';
    if (bVal) bVal.textContent = Utils.fmt(value, 3);

    if (bInd) {
        if (mode === 'drain') {
            bInd.textContent = '⚡ Reset';
            bInd.style.fill = '#ef4444';
            bf.style.filter = 'drop-shadow(0 0 8px rgba(239,68,68,0.6))';
        } else if (mode === 'charge') {
            bInd.textContent = '🔋 Update';
            bInd.style.fill = '#22c55e';
            bf.style.filter = 'drop-shadow(0 0 8px rgba(34,197,94,0.6))';
        } else if (mode === 'final') {
            bInd.textContent = '✓ h(t)';
            bInd.style.fill = 'var(--text-secondary)';
            bf.style.filter = 'none';
        } else {
            bInd.textContent = '';
            bf.style.filter = 'none';
        }
    }
}

// ==================== ANIMATION SUB-STEPS ====================
async function animateSubStep(step) {
    const state = demoData.cell_states[currentCell];
    const dur = anim.getStepDuration();
    const stepInfo = $('svg-step-info');
    if (stepInfo) stepInfo.textContent = `Bước ${step}/${TOTAL_SUB_STEPS}`;

    switch (step) {
        case 1: await animStep1(state, dur); break;
        case 2: await animStep2(state, dur); break;
        case 3: await animStep3(state, dur); break;
        case 4: await animStep4(state, dur); break;
        case 5: await animStep5(state, dur); break;
    }
}

async function animStep1(state, dur) {
    // Input arrives: x(t) + h(t-1) concat
    $('pipe-input-x')?.classList.add('active-candidate');
    $('pipe-ht-in')?.classList.add('active-hidden');
    await Utils.sleep(dur * 0.5);
    const c = $('concat-point');
    if (c) { c.style.fill = 'var(--accent-primary)'; c.style.filter = 'drop-shadow(0 0 12px var(--accent-primary))'; }
    await Utils.sleep(dur * 0.5);
}

async function animStep2(state, dur) {
    // Reset Gate
    $('pipe-to-reset')?.classList.add('active-forget');
    await Utils.sleep(dur * 0.2);
    $('gate-bg-reset')?.classList.add('active');
    const v = $('val-reset'); if (v) { v.textContent = Utils.fmt(state.reset_gate.mean, 3); v.classList.add('visible'); }
    await Utils.sleep(dur * 0.3);
    $('pipe-reset-up')?.classList.add('active-forget');
    const op = $('op-reset-mul'); if (op) { op.style.stroke = 'var(--reset-gate)'; op.style.filter = 'drop-shadow(0 0 10px var(--reset-gate))'; }
    $('pipe-ht-to-reset-mul')?.classList.add('active-hidden');

    // ★ Battery DRAIN: reset gate "filters" h(t-1)
    const prevH = currentCell > 0 ? demoData.cell_states[currentCell - 1].hidden_state.mean : 0;
    const afterReset = prevH * state.reset_gate.mean;
    updateBattery(afterReset, 'drain');
    await Utils.sleep(dur * 0.5);
}

async function animStep3(state, dur) {
    // Update Gate
    $('pipe-to-update')?.classList.add('active-output');
    await Utils.sleep(dur * 0.2);
    $('gate-bg-update')?.classList.add('active');
    const v = $('val-update'); if (v) { v.textContent = Utils.fmt(state.update_gate.mean, 3); v.classList.add('visible'); }
    await Utils.sleep(dur * 0.3);
    $('pipe-update-to-1mz')?.classList.add('active-output');
    $('pipe-update-to-z')?.classList.add('active-output');
    await Utils.sleep(dur * 0.5);
}

async function animStep4(state, dur) {
    // Candidate — x(t) branch (cyan) + reset×h(t-1) branch (orange)
    $('pipe-to-cand')?.classList.add('active-candidate');
    $('pipe-reset-to-cand')?.classList.add('active-forget');  // orange to distinguish from x(t)
    await Utils.sleep(dur * 0.3);
    $('gate-bg-candidate')?.classList.add('active');
    const v = $('val-candidate'); if (v) { v.textContent = Utils.fmt(state.candidate.mean, 3); v.classList.add('visible'); }
    await Utils.sleep(dur * 0.3);
    $('pipe-cand-up')?.classList.add('active-candidate');
    const op = $('op-z-mul'); if (op) { op.style.stroke = 'var(--update-gate)'; op.style.filter = 'drop-shadow(0 0 10px var(--update-gate))'; }
    await Utils.sleep(dur * 0.4);
}

async function animStep5(state, dur) {
    // Hidden state update: h(t) = (1-z)·h(t-1) + z·h̃(t)
    $('pipe-ht-to-1mz')?.classList.add('active-hidden');
    const op1mz = $('op-1mz-mul'); if (op1mz) { op1mz.style.stroke = 'var(--hidden-state)'; op1mz.style.filter = 'drop-shadow(0 0 10px var(--hidden-state))'; }
    await Utils.sleep(dur * 0.3);
    const opAdd = $('op-add'); if (opAdd) { opAdd.style.stroke = 'var(--hidden-state)'; opAdd.style.filter = 'drop-shadow(0 0 14px var(--hidden-state))'; }
    await Utils.sleep(dur * 0.2);

    // ★ Battery CHARGE: new hidden state
    updateBattery(state.hidden_state.mean, 'charge');
    await Utils.sleep(dur * 0.2);

    $('hs-lane')?.classList.add('active');
    updateBattery(state.hidden_state.mean, 'final');
    await Utils.sleep(dur * 0.3);
}

// ==================== INSTANT APPLY ====================
function applySubStepInstant(step) {
    const state = demoData.cell_states[currentCell];
    if (!state) return;
    switch (step) {
        case 1: {
            $('pipe-input-x')?.classList.add('active-candidate');
            $('pipe-ht-in')?.classList.add('active-hidden');
            const c = $('concat-point');
            if (c) { c.style.fill = 'var(--accent-primary)'; c.style.filter = 'drop-shadow(0 0 12px var(--accent-primary))'; }
            break;
        }
        case 2: {
            $('pipe-to-reset')?.classList.add('active-forget');
            $('gate-bg-reset')?.classList.add('active');
            $('pipe-reset-up')?.classList.add('active-forget');
            const op = $('op-reset-mul'); if (op) { op.style.stroke = 'var(--reset-gate)'; op.style.filter = 'drop-shadow(0 0 10px var(--reset-gate))'; }
            $('pipe-ht-to-reset-mul')?.classList.add('active-hidden');
            const v = $('val-reset'); if (v) { v.textContent = Utils.fmt(state.reset_gate.mean, 3); v.classList.add('visible'); }
            const prevH2 = currentCell > 0 ? demoData.cell_states[currentCell - 1].hidden_state.mean : 0;
            updateBattery(prevH2 * state.reset_gate.mean, 'drain');
            break;
        }
        case 3: {
            $('pipe-to-update')?.classList.add('active-output');
            $('gate-bg-update')?.classList.add('active');
            $('pipe-update-to-1mz')?.classList.add('active-output');
            $('pipe-update-to-z')?.classList.add('active-output');
            const v = $('val-update'); if (v) { v.textContent = Utils.fmt(state.update_gate.mean, 3); v.classList.add('visible'); }
            break;
        }
        case 4: {
            $('pipe-to-cand')?.classList.add('active-candidate');
            $('pipe-reset-to-cand')?.classList.add('active-forget');  // orange
            $('gate-bg-candidate')?.classList.add('active');
            $('pipe-cand-up')?.classList.add('active-candidate');
            const op = $('op-z-mul'); if (op) { op.style.stroke = 'var(--update-gate)'; op.style.filter = 'drop-shadow(0 0 10px var(--update-gate))'; }
            const v = $('val-candidate'); if (v) { v.textContent = Utils.fmt(state.candidate.mean, 3); v.classList.add('visible'); }
            break;
        }
        case 5: {
            $('pipe-ht-to-1mz')?.classList.add('active-hidden');
            const op1mz = $('op-1mz-mul'); if (op1mz) { op1mz.style.stroke = 'var(--hidden-state)'; op1mz.style.filter = 'drop-shadow(0 0 10px var(--hidden-state))'; }
            const opAdd = $('op-add'); if (opAdd) { opAdd.style.stroke = 'var(--hidden-state)'; opAdd.style.filter = 'drop-shadow(0 0 14px var(--hidden-state))'; }
            $('hs-lane')?.classList.add('active');
            updateBattery(state.hidden_state.mean, 'final');
            break;
        }
    }
}

// ==================== TOOLTIPS ====================
function showTooltip(event, gateType) {
    const tooltip = document.getElementById('gate-tooltip');
    const state = tooltipData;
    if (!state || !state.formulas) return;
    const info = {
        reset: { icon: '🟠', name: 'Reset Gate (Cổng Đặt Lại)', desc: 'Quyết định bao nhiêu thông tin từ hidden state trước đó sẽ được "reset".\nGiá trị gần 0 → reset nhiều, gần 1 → giữ nguyên.', formula: state.formulas.reset, value: state.reset_gate.mean, interpret: v => `→ Giữ lại ${(v * 100).toFixed(0)}% thông tin cũ cho candidate` },
        update: { icon: '🟣', name: 'Update Gate (Cổng Cập Nhật)', desc: 'Quyết định tỷ lệ giữ thông tin cũ h(t-1) và thêm thông tin mới h̃(t).\nz gần 1 → ưu tiên mới, z gần 0 → giữ cũ.', formula: state.formulas.update, value: state.update_gate.mean, interpret: v => `→ ${(v * 100).toFixed(0)}% mới + ${((1-v) * 100).toFixed(0)}% cũ` },
        candidate: { icon: '🔵', name: 'Candidate (Hidden State Ứng Viên)', desc: 'Tạo hidden state mới dựa trên input và phần hidden state đã được reset.\nSử dụng tanh (giá trị -1 đến 1).', formula: state.formulas.candidate, value: state.candidate.mean, interpret: v => `→ Giá trị trung bình: ${Utils.fmt(v, 3)}` }
    };
    const gate = info[gateType];
    if (!gate) return;
    document.getElementById('tooltip-title').innerHTML = `${gate.icon} ${gate.name}`;
    document.getElementById('tooltip-desc').textContent = gate.desc;
    document.getElementById('tooltip-formula').textContent = gate.formula;
    document.getElementById('tooltip-value').textContent = `Giá trị: ${Utils.fmt(gate.value, 4)}`;
    document.getElementById('tooltip-interpret').textContent = gate.interpret(gate.value);
    const rect = event.target.closest('.gate-node').getBoundingClientRect();
    tooltip.style.left = Math.min(rect.left, window.innerWidth - 400) + 'px';
    tooltip.style.top = (rect.bottom + 10) + 'px';
    tooltip.classList.add('visible');
}
function hideTooltip() { document.getElementById('gate-tooltip').classList.remove('visible'); }

// ==================== OUTPUT ====================
async function showOutput() {
    const section = document.getElementById('output-section');
    const pred = demoData.prediction;

    section.classList.add('active');
    section.scrollIntoView({ behavior: 'smooth', block: 'center' });
    await Utils.sleep(600);

    const barPos = document.getElementById('bar-positive');
    const pctPos = document.getElementById('pct-positive');
    const posPct = pred.probabilities.positive * 100;
    await animateBar(barPos, pctPos, posPct, 800);
    await Utils.sleep(300);

    const barNeg = document.getElementById('bar-negative');
    const pctNeg = document.getElementById('pct-negative');
    const negPct = pred.probabilities.negative * 100;
    await animateBar(barNeg, pctNeg, negPct, 800);
    await Utils.sleep(500);

    const conclusion = document.getElementById('prediction-conclusion');
    conclusion.style.opacity = '0';
    conclusion.style.transform = 'scale(0.8) translateY(20px)';
    conclusion.style.transition = 'none';

    if (pred.label === 'Tích cực') {
        conclusion.className = 'prediction-conclusion positive';
        conclusion.innerHTML = `😊 Câu này mang sắc thái <b>TÍCH CỰC</b> (${Utils.pct(pred.confidence)})`;
    } else {
        conclusion.className = 'prediction-conclusion negative';
        conclusion.innerHTML = `😔 Câu này mang sắc thái <b>TIÊU CỰC</b> (${Utils.pct(pred.confidence)})`;
    }

    await Utils.sleep(100);
    conclusion.style.transition = 'all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)';
    conclusion.style.opacity = '1';
    conclusion.style.transform = 'scale(1) translateY(0)';

    await Utils.sleep(600);
    conclusion.style.boxShadow = pred.label === 'Tích cực'
        ? '0 0 30px rgba(34, 197, 94, 0.5), 0 0 60px rgba(34, 197, 94, 0.2)'
        : '0 0 30px rgba(239, 68, 68, 0.5), 0 0 60px rgba(239, 68, 68, 0.2)';
    await Utils.sleep(1000);
    conclusion.style.boxShadow = '';

    anim.stopAutoPlay();
    document.getElementById('btn-autoplay').textContent = '▶▶ Tự Động';
}

function animateBar(barEl, pctEl, targetPct, duration) {
    return new Promise(resolve => {
        const t0 = performance.now();
        function tick(now) {
            const p = Math.min((now - t0) / duration, 1);
            const eased = 1 - Math.pow(1 - p, 3);
            const current = targetPct * eased;
            barEl.style.width = current + '%';
            pctEl.textContent = current.toFixed(1) + '%';
            if (p < 1) requestAnimationFrame(tick);
            else { pctEl.textContent = targetPct.toFixed(1) + '%'; resolve(); }
        }
        requestAnimationFrame(tick);
    });
}
