/**
 * LSTM Demo V2.2 — Fixed centering with simple flex layout
 * Active cell is ALWAYS centered via CSS flex justify-content:center
 * Mini previews shown on left (done) and right (future)
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

// ==================== SCOPED ELEMENT QUERY ====================
function $(id) {
    const container = document.getElementById('active-cell-svg');
    if (container) {
        return container.querySelector('#' + id);
    }
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
        demoData = await API.predictLSTM(text);
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
    await flyTokenToCell(currentCell, { force: true });
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

    // Left side: done cells (show last 2 max)
    leftContainer.innerHTML = '';
    const leftStart = Math.max(0, currentCell - 2);
    for (let i = leftStart; i < currentCell; i++) {
        const mini = createMiniPreview(i, true);
        leftContainer.appendChild(mini);
    }

    // Right side: future cells (show next 4 max)
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

    // Left connector: show if not first cell
    if (currentCell > 0) {
        connLeft.style.display = '';
        const prevState = demoData.cell_states[currentCell - 1];
        if (prevState) {
            connLeftVals.innerHTML = `
                <div>h: ${Utils.fmt(prevState.hidden_state.mean, 3)}</div>
                <div>C: ${Utils.fmt(prevState.cell_state.mean, 3)}</div>
            `;
            connLeftVals.classList.add('has-value');
        }
    } else {
        connLeft.style.display = 'none';
    }

    // Right connector: keep visible on last cell to show final output states
    if (currentCell < totalCells - 1) {
        connRight.style.display = '';
        connRightVals.innerHTML = '<div>h: —</div><div>C: —</div>';
        connRightVals.classList.remove('has-value');
    } else {
        connRight.style.display = '';
        const currState = demoData.cell_states[currentCell];
        if (currState) {
            connRightVals.innerHTML = `
                <div>h(T): ${Utils.fmt(currState.hidden_state.mean, 3)}</div>
                <div>C(T): ${Utils.fmt(currState.cell_state.mean, 3)}</div>
            `;
            connRightVals.classList.add('has-value');
        } else {
            connRightVals.innerHTML = '<div>h(T): —</div><div>C(T): —</div>';
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
        `Cell ${currentCell + 1}/${totalCells} · ${currentSubStep}/7`;
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

    if (currentSubStep < 7) {
        currentSubStep++;
        await animateSubStep(currentSubStep);
    }

    if (currentSubStep >= 7) {
        // ★ Delay to let user review completed cell output
        await Utils.sleep(2000 / anim.speed);

        currentCell++;
        currentSubStep = 0;
        if (currentCell < totalCells) {
            // Render new cell view (minis, connectors, SVG)
            const activeEl = document.getElementById('chain-active');
            activeEl.style.opacity = '0';
            await Utils.sleep(300);
            renderCellView();
            activeEl.style.opacity = '1';

            // ★ Fly token/vector from tokens row down to cell
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
        currentSubStep = 6;
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
async function flyTokenToCell(idx, options = {}) {
    const { force = false } = options;
    const vecEl = document.getElementById(`te-vector-${idx}`);
    const cellContainer = document.getElementById('active-cell-svg');
    if (!vecEl || !cellContainer) return;
    if (!force && idx === 0 && currentSubStep === 0) return;

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
    const startY = vecRect.top + window.scrollY;  // absolute page Y

    flyVecText.textContent = Utils.summarizeVector(demoData.embeddings[idx].summary);
    flyVec.style.display = '';
    flyVec.style.position = 'absolute'; // use absolute positioning (not fixed) so it works with scroll
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
        // Calculate target: x(t) at SVG coords (90, 430)
        const svgRect = svg.getBoundingClientRect();
        const scaleX = svgRect.width / 760;
        const scaleY = svgRect.height / 450;
        const endX = svgRect.left + (90 * scaleX) - 30;
        const endY = (svgRect.top + window.scrollY) + (430 * scaleY) - 10; // absolute page Y

        const duration = 2000; // 2 seconds for smooth descent

        await new Promise(resolve => {
            const t0 = performance.now();

            function easeInOutCubic(t) {
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            }

            function tick(now) {
                const elapsed = now - t0;
                const rawProgress = Math.min(elapsed / duration, 1);
                const p = easeInOutCubic(rawProgress);

                // Interpolate vector position (absolute page coords)
                const curX = startX + (endX - startX) * p;
                const curY = startY + (endY - startY) * p;

                flyVec.style.left = curX + 'px';
                flyVec.style.top = curY + 'px';

                // Scroll so the vector stays at ~45% of viewport height
                const targetViewportY = window.innerHeight * 0.45;
                const desiredScrollY = curY - targetViewportY;
                window.scrollTo({ top: desiredScrollY, behavior: 'auto' });

                if (rawProgress < 1) {
                    requestAnimationFrame(tick);
                } else {
                    resolve();
                }
            }

            requestAnimationFrame(tick);
        });

        // ★ Flash when vector arrives at x(t)
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
        flyVec.style.position = 'fixed'; // restore
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

    document.getElementById('active-cell-title').textContent = `Cell t=${currentCell + 1}`;
    document.getElementById('active-cell-token').innerHTML = `"${state.token}"`;
    tooltipData = state;

    const prevState = currentCell > 0 ? demoData.cell_states[currentCell - 1] : null;
    const prevC = prevState ? Utils.fmt(prevState.cell_state.mean, 2) : '0.00';
    const prevH = prevState ? Utils.fmt(prevState.hidden_state.mean, 2) : '0.00';

    const container = document.getElementById('active-cell-svg');
    const W = 760, H = 450;
    container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" class="cell-svg" xmlns="http://www.w3.org/2000/svg">
        <!-- CELL STATE LANE (y=55) -->
        <text x="10" y="45" class="state-label" fill="var(--cell-state)">C(t-1)=${prevC}</text>
        <line x1="50" y1="55" x2="680" y2="55" class="state-lane cell-state-lane active" id="cs-lane"/>
        <polygon points="680,49 696,55 680,61" fill="var(--cell-state)" opacity="0.3" id="cs-arrow"/>
        <text x="598" y="43" class="state-label" fill="var(--cell-state)">C(t)=</text>
        <text x="640" y="43" class="state-label" fill="var(--cell-state)" id="val-current-c">—</text>

        <!-- BATTERY BAR (enhanced with % and drain/charge indicator) -->
        <g id="cell-battery" transform="translate(708, 56)">
            <text x="20" y="-8" class="battery-label" text-anchor="middle">Cell State</text>
            <!-- Battery cap -->
            <rect x="10" y="4" width="20" height="5" rx="2" fill="var(--text-muted)" opacity="0.4"/>
            <!-- Battery body -->
            <rect x="2" y="9" width="36" height="100" rx="4" class="battery-outer"/>
            <!-- Battery fill (starts from bottom) -->
            <rect x="5" y="12" width="30" height="94" id="battery-fill" class="battery-fill" rx="2"/>
            <!-- Percentage text -->
            <text x="20" y="65" class="battery-pct" id="battery-pct" text-anchor="middle">0%</text>
            <!-- Value text below -->
            <text x="20" y="124" class="battery-value" id="battery-value" text-anchor="middle">0.00</text>
            <!-- Drain/Charge indicator -->
            <text x="20" y="140" class="battery-indicator" id="battery-indicator" text-anchor="middle"></text>
        </g>

        <!-- HIDDEN STATE LANE (y=380) -->
        <text x="10" y="373" class="state-label" fill="var(--hidden-state)">h(t-1)=${prevH}</text>
        <line x1="50" y1="380" x2="680" y2="380" class="state-lane hidden-state-lane" id="hs-lane"/>
        <polygon points="680,374 696,380 680,386" fill="var(--hidden-state)" opacity="0.3"/>
        <text x="620" y="373" class="state-label" fill="var(--hidden-state)">h(t)=</text>
        <text x="662" y="373" class="state-label" fill="var(--hidden-state)" id="val-current-h">—</text>

        <!-- OPERATORS ON CELL STATE -->
        <circle cx="190" cy="55" r="18" class="operator-bg" id="op-forget-mul"/>
        <text x="190" y="55" class="operator-text" font-size="18">×</text>
        <circle cx="435" cy="55" r="18" class="operator-bg" id="op-add"/>
        <text x="435" y="55" class="operator-text" font-size="18">+</text>

        <!-- INTERMEDIATE OPERATORS -->
        <circle cx="340" cy="170" r="16" class="operator-bg" id="op-input-mul"/>
        <text x="340" y="170" class="operator-text" font-size="16">×</text>
        <rect x="535" y="120" width="56" height="30" rx="12" class="operator-bg" id="op-tanh-cs"/>
        <text x="563" y="138" class="activation-text tanh-fn" font-size="14">tanh</text>
        <circle cx="630" cy="200" r="16" class="operator-bg" id="op-output-mul"/>
        <text x="630" y="200" class="operator-text" font-size="16">×</text>

        <!-- PIPES -->
        <path d="M50,380 L90,380 L90,350" class="pipe" id="pipe-ht-in"/>
        <path d="M90,435 L90,350" class="pipe" id="pipe-input-x"/>
        <text x="105" y="435" class="cell-label" fill="var(--accent-primary)" font-size="12" font-weight="600">x(t)</text>

        <line x1="105" y1="340" x2="560" y2="340" class="pipe" id="pipe-concat-bus" stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="4,4" opacity="0.15"/>

        <path d="M105,340 L190,340 L190,300" class="pipe" id="pipe-to-forget"/>
        <path d="M105,340 L300,340 L300,300" class="pipe" id="pipe-to-input"/>
        <path d="M105,340 L385,340 L385,300" class="pipe" id="pipe-to-candidate"/>
        <path d="M105,340 L540,340 L540,300" class="pipe" id="pipe-to-output"/>

        <path d="M190,240 L190,73" class="pipe" id="pipe-forget-up"/>
        <path d="M300,240 L300,170 L324,170" class="pipe" id="pipe-input-up"/>
        <path d="M385,240 L385,170 L356,170" class="pipe" id="pipe-candidate-up"/>
        <path d="M340,154 L340,100 L435,100 L435,73" class="pipe" id="pipe-ic-up"/>
        <path d="M560,55 L560,120" class="pipe" id="pipe-cs-tanh"/>
        <path d="M591,135 L630,135 L630,184" class="pipe" id="pipe-tanh-to-outmul"/>
        <path d="M540,240 L540,200 L614,200" class="pipe" id="pipe-output-up"/>
        <path d="M630,216 L630,380" class="pipe" id="pipe-out-hidden"/>

        <!-- CONCAT POINT -->
        <circle cx="90" cy="340" r="12" fill="var(--bg-tertiary)" stroke="var(--border-medium)" stroke-width="1.5" id="concat-point"/>
        <text x="90" y="345" text-anchor="middle" dominant-baseline="central" font-size="9" fill="var(--text-secondary)">[,]</text>
        <text x="57" y="330" class="cell-label" fill="var(--text-muted)" font-size="9">concat</text>

        <!-- GATES -->
        <g class="gate-node" id="gate-forget" onmouseenter="showTooltip(event,'forget')" onmouseleave="hideTooltip()">
            <rect x="155" y="243" width="70" height="55" class="gate-bg forget" id="gate-bg-forget"/>
            <text x="190" y="259" class="gate-label">Forget</text>
            <text x="190" y="273" class="gate-sublabel" fill="var(--forget-gate)">f(t)</text>
            <circle cx="225" cy="243" r="12" fill="rgba(255,107,107,0.2)" stroke="var(--forget-gate)" stroke-width="1.5"/>
            <text x="225" y="247" text-anchor="middle" fill="var(--forget-gate)" font-family="'JetBrains Mono', monospace" font-size="14" font-weight="700">σ</text>
            <text x="190" y="315" class="gate-value" id="val-forget"></text>
        </g>
        <g class="gate-node" id="gate-input" onmouseenter="showTooltip(event,'input')" onmouseleave="hideTooltip()">
            <rect x="265" y="243" width="70" height="55" class="gate-bg input" id="gate-bg-input"/>
            <text x="300" y="259" class="gate-label">Input</text>
            <text x="300" y="273" class="gate-sublabel" fill="var(--input-gate)">i(t)</text>
            <circle cx="335" cy="243" r="12" fill="rgba(38,222,129,0.2)" stroke="var(--input-gate)" stroke-width="1.5"/>
            <text x="335" y="247" text-anchor="middle" fill="var(--input-gate)" font-family="'JetBrains Mono', monospace" font-size="14" font-weight="700">σ</text>
            <text x="300" y="315" class="gate-value" id="val-input"></text>
        </g>
        <g class="gate-node" id="gate-candidate" onmouseenter="showTooltip(event,'candidate')" onmouseleave="hideTooltip()">
            <rect x="350" y="243" width="70" height="55" class="gate-bg candidate" id="gate-bg-candidate"/>
            <text x="385" y="259" class="gate-label">Candi.</text>
            <text x="385" y="273" class="gate-sublabel" fill="var(--candidate)">C̃(t)</text>
            <rect x="395" y="231" width="36" height="18" rx="9" fill="rgba(69,183,209,0.2)" stroke="var(--candidate)" stroke-width="1.5"/>
            <text x="413" y="243" text-anchor="middle" fill="var(--candidate)" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="700">tanh</text>
            <text x="385" y="315" class="gate-value" id="val-candidate"></text>
        </g>
        <g class="gate-node" id="gate-output" onmouseenter="showTooltip(event,'output')" onmouseleave="hideTooltip()">
            <rect x="505" y="243" width="70" height="55" class="gate-bg output" id="gate-bg-output"/>
            <text x="540" y="259" class="gate-label">Output</text>
            <text x="540" y="273" class="gate-sublabel" fill="var(--output-gate)">o(t)</text>
            <circle cx="575" cy="243" r="12" fill="rgba(165,94,234,0.2)" stroke="var(--output-gate)" stroke-width="1.5"/>
            <text x="575" y="247" text-anchor="middle" fill="var(--output-gate)" font-family="'JetBrains Mono', monospace" font-size="14" font-weight="700">σ</text>
            <text x="540" y="315" class="gate-value" id="val-output"></text>
        </g>

        <!-- Annotations -->
        <text x="190" y="93" class="annotation-text" fill="var(--forget-gate)">f(t)·C(t-1)</text>
        <text x="340" y="147" class="annotation-text" fill="var(--input-gate)">i(t)·C̃(t)</text>
        <text x="630" y="178" class="annotation-text" fill="var(--output-gate)">o(t)·tanh(C)</text>

        <text x="${W-118}" y="18" class="step-indicator" id="svg-step-info">Bước ${currentSubStep}/7</text>
    </svg>`;

    // Set initial battery level based on C(t-1)
    const prevCVal = prevState ? prevState.cell_state.mean : 0;
    const prevHVal = prevState ? prevState.hidden_state.mean : 0;
    setTimeout(() => {
        updateBattery(prevCVal, 'init');
        updateStateOutputValues(prevCVal, null);
        if (currentCell === 0) {
            updateStateOutputValues(null, prevHVal);
        }
    }, 100);
}

// ==================== BATTERY HELPER ====================
function updateBattery(value, mode) {
    // value: cell state mean (-1 to 1), mode: 'init'|'drain'|'charge'|'final'
    const bf = $('battery-fill');
    const bPct = $('battery-pct');
    const bVal = $('battery-value');
    const bInd = $('battery-indicator');
    if (!bf) return;

    // Sigmoid scaling: maps cell state mean to 0-100%
    // value=0 → 50%, positive → >50%, negative → <50%
    // Factor 10 amplifies small mean values for visible difference
    const sigmoid = 1 / (1 + Math.exp(-value * 10));
    const pct = Math.round(sigmoid * 100);
    const fillRatio = sigmoid; // 0.0 to 1.0

    // Fill height: 94px max, grows from bottom
    const fillH = Math.max(2, 94 * fillRatio);
    const fillY = 12 + (94 - fillH);

    // Color based on level (red=low/negative, green=high/positive)
    let fillColor;
    if (pct < 30) fillColor = '#ef4444';        // red — very negative
    else if (pct < 45) fillColor = '#f59e0b';    // amber — slightly negative
    else if (pct < 55) fillColor = '#eab308';    // yellow — neutral
    else if (pct < 70) fillColor = '#22c55e';    // green — slightly positive
    else fillColor = '#10b981';                   // emerald — very positive

    // Apply with transition
    bf.style.transition = 'all 0.8s ease';
    bf.setAttribute('y', fillY.toFixed(0));
    bf.setAttribute('height', fillH.toFixed(0));
    bf.style.fill = fillColor;
    bf.style.opacity = '0.75';

    if (bPct) bPct.textContent = pct + '%';
    if (bVal) bVal.textContent = Utils.fmt(value, 3);

    // Mode-specific effects
    if (bInd) {
        if (mode === 'drain') {
            bInd.textContent = '⚡ Quên';
            bInd.style.fill = '#ef4444';
            bf.style.filter = 'drop-shadow(0 0 8px rgba(239,68,68,0.6))';
        } else if (mode === 'charge') {
            bInd.textContent = '🔋 Nạp';
            bInd.style.fill = '#22c55e';
            bf.style.filter = 'drop-shadow(0 0 8px rgba(34,197,94,0.6))';
        } else if (mode === 'final') {
            bInd.textContent = '✓ C(t)';
            bInd.style.fill = 'var(--text-secondary)';
            bf.style.filter = 'none';
        } else {
            bInd.textContent = '';
            bf.style.filter = 'none';
        }
    }
}

function updateStateOutputValues(cVal = null, hVal = null) {
    const cText = $('val-current-c');
    const hText = $('val-current-h');
    if (cText && cVal !== null && cVal !== undefined) {
        cText.textContent = Utils.fmt(cVal, 3);
        cText.style.fill = 'var(--cell-state)';
        cText.style.fontWeight = '700';
        cText.style.opacity = '1';
    }
    if (hText && hVal !== null && hVal !== undefined) {
        hText.textContent = Utils.fmt(hVal, 3);
        hText.style.fill = 'var(--hidden-state)';
        hText.style.fontWeight = '700';
        hText.style.opacity = '1';
    }
}

// ==================== ANIMATION SUB-STEPS ====================
async function animateSubStep(step) {
    const state = demoData.cell_states[currentCell];
    const dur = anim.getStepDuration();
    const stepInfo = $('svg-step-info');
    if (stepInfo) stepInfo.textContent = `Bước ${step}/7`;

    switch (step) {
        case 1: await animStep1(state, dur); break;
        case 2: await animStep2(state, dur); break;
        case 3: await animStep3(state, dur); break;
        case 4: await animStep4(state, dur); break;
        case 5: await animStep5(state, dur); break;
        case 6: await animStep6(state, dur); break;
        case 7: await animStep7(state, dur); break;
    }
}

async function animStep1(state, dur) {
    $('pipe-ht-in')?.classList.add('active-hidden');
    await Utils.sleep(dur * 0.3);
    $('pipe-input-x')?.classList.add('active-candidate');
    await Utils.sleep(dur * 0.3);
    const c = $('concat-point'); if (c) { c.style.fill = 'var(--accent-primary)'; c.style.filter = 'drop-shadow(0 0 12px var(--accent-primary))'; }
    const bus = $('pipe-concat-bus'); if (bus) { bus.style.opacity = '0.6'; bus.style.stroke = 'var(--accent-primary)'; }
    await Utils.sleep(dur * 0.4);
}

async function animStep2(state, dur) {
    // Forget gate: pipe → gate → battery DRAIN
    $('pipe-to-forget')?.classList.add('active-forget');
    await Utils.sleep(dur * 0.2);
    $('gate-bg-forget')?.classList.add('active');
    const v = $('val-forget'); if (v) { v.textContent = Utils.fmt(state.forget_gate.mean, 3); v.classList.add('visible'); }
    await Utils.sleep(dur * 0.3);
    $('pipe-forget-up')?.classList.add('active-forget');
    const op = $('op-forget-mul'); if (op) { op.style.stroke = 'var(--forget-gate)'; op.style.filter = 'drop-shadow(0 0 10px var(--forget-gate))'; }

    // ★ Battery DRAIN: C(t-1) * f(t) — forget gate reduces cell state
    const prevC = currentCell > 0 ? demoData.cell_states[currentCell - 1].cell_state.mean : 0;
    const afterForget = prevC * state.forget_gate.mean;
    updateBattery(afterForget, 'drain');
    await Utils.sleep(dur * 0.5);
}

async function animStep3(state, dur) {
    $('pipe-to-input')?.classList.add('active-input');
    await Utils.sleep(dur * 0.25);
    $('gate-bg-input')?.classList.add('active');
    const v = $('val-input'); if (v) { v.textContent = Utils.fmt(state.input_gate.mean, 3); v.classList.add('visible'); }
    $('pipe-input-up')?.classList.add('active-input');
    await Utils.sleep(dur * 0.75);
}

async function animStep4(state, dur) {
    $('pipe-to-candidate')?.classList.add('active-candidate');
    await Utils.sleep(dur * 0.2);
    $('gate-bg-candidate')?.classList.add('active');
    const v = $('val-candidate'); if (v) { v.textContent = Utils.fmt(state.candidate.mean, 3); v.classList.add('visible'); }
    $('pipe-candidate-up')?.classList.add('active-candidate');
    await Utils.sleep(dur * 0.25);
    const op = $('op-input-mul'); if (op) { op.style.stroke = 'var(--input-gate)'; op.style.filter = 'drop-shadow(0 0 10px var(--input-gate))'; }
    await Utils.sleep(dur * 0.2);
    $('pipe-ic-up')?.classList.add('active-candidate');
    await Utils.sleep(dur * 0.35);
}

async function animStep5(state, dur) {
    // Cell state update: C(t) = f(t)*C(t-1) + i(t)*candidate
    const opAdd = $('op-add'); if (opAdd) { opAdd.style.stroke = 'var(--cell-state)'; opAdd.style.filter = 'drop-shadow(0 0 14px var(--cell-state))'; }
    await Utils.sleep(dur * 0.3);
    $('cs-lane')?.classList.add('active');
    $('cs-lane')?.classList.add('pulse');

    // ★ Battery CHARGE: new info added → cell state increases
    updateBattery(state.cell_state.mean, 'charge');
    updateStateOutputValues(state.cell_state.mean, null);
    await Utils.sleep(dur * 0.4);

    // ★ Final value settles
    updateBattery(state.cell_state.mean, 'final');
    updateStateOutputValues(state.cell_state.mean, null);
    await Utils.sleep(dur * 0.3);
}
async function animStep6(state, dur) {
    $('pipe-to-output')?.classList.add('active-output');
    await Utils.sleep(dur * 0.25);
    $('gate-bg-output')?.classList.add('active');
    const v = $('val-output'); if (v) { v.textContent = Utils.fmt(state.output_gate.mean, 3); v.classList.add('visible'); }
    $('pipe-output-up')?.classList.add('active-output');
    await Utils.sleep(dur * 0.75);
}
async function animStep7(state, dur) {
    $('pipe-cs-tanh')?.classList.add('active-cell');
    const oT = $('op-tanh-cs'); if (oT) { oT.style.stroke = 'var(--candidate)'; oT.style.filter = 'drop-shadow(0 0 10px var(--candidate))'; }
    await Utils.sleep(dur * 0.25);
    $('pipe-tanh-to-outmul')?.classList.add('active-cell');
    const oM = $('op-output-mul'); if (oM) { oM.style.stroke = 'var(--output-gate)'; oM.style.filter = 'drop-shadow(0 0 12px var(--output-gate))'; }
    await Utils.sleep(dur * 0.25);
    $('pipe-out-hidden')?.classList.add('active-hidden');
    await Utils.sleep(dur * 0.2);
    $('hs-lane')?.classList.add('active');
    $('cs-lane')?.classList.add('active');
    $('cs-lane')?.classList.add('pulse');
    updateBattery(state.cell_state.mean, 'final');
    updateStateOutputValues(state.cell_state.mean, state.hidden_state.mean);
    updateConnectors();
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
            const c = $('concat-point'); if (c) { c.style.fill = 'var(--accent-primary)'; c.style.filter = 'drop-shadow(0 0 12px var(--accent-primary))'; }
            const bus = $('pipe-concat-bus'); if (bus) { bus.style.opacity = '0.6'; bus.style.stroke = 'var(--accent-primary)'; }
            break;
        }
        case 2: {
            $('pipe-to-forget')?.classList.add('active-forget');
            $('gate-bg-forget')?.classList.add('active');
            $('pipe-forget-up')?.classList.add('active-forget');
            const op = $('op-forget-mul'); if (op) { op.style.stroke = 'var(--forget-gate)'; op.style.filter = 'drop-shadow(0 0 10px var(--forget-gate))'; }
            const v = $('val-forget'); if (v) { v.textContent = Utils.fmt(state.forget_gate.mean, 3); v.classList.add('visible'); }
            const prevC2 = currentCell > 0 ? demoData.cell_states[currentCell - 1].cell_state.mean : 0;
            updateBattery(prevC2 * state.forget_gate.mean, 'drain');
            break;
        }
        case 3: {
            $('pipe-to-input')?.classList.add('active-input');
            $('gate-bg-input')?.classList.add('active');
            $('pipe-input-up')?.classList.add('active-input');
            const v = $('val-input'); if (v) { v.textContent = Utils.fmt(state.input_gate.mean, 3); v.classList.add('visible'); }
            break;
        }
        case 4: {
            $('pipe-to-candidate')?.classList.add('active-candidate');
            $('gate-bg-candidate')?.classList.add('active');
            $('pipe-candidate-up')?.classList.add('active-candidate');
            const op = $('op-input-mul'); if (op) { op.style.stroke = 'var(--input-gate)'; op.style.filter = 'drop-shadow(0 0 10px var(--input-gate))'; }
            $('pipe-ic-up')?.classList.add('active-candidate');
            const v = $('val-candidate'); if (v) { v.textContent = Utils.fmt(state.candidate.mean, 3); v.classList.add('visible'); }
            break;
        }
        case 5: {
            const opAdd = $('op-add'); if (opAdd) { opAdd.style.stroke = 'var(--cell-state)'; opAdd.style.filter = 'drop-shadow(0 0 14px var(--cell-state))'; }
            $('cs-lane')?.classList.add('active');
            updateBattery(state.cell_state.mean, 'final');
            updateStateOutputValues(state.cell_state.mean, null);
            break;
        }
        case 6: {
            $('pipe-to-output')?.classList.add('active-output');
            $('gate-bg-output')?.classList.add('active');
            $('pipe-output-up')?.classList.add('active-output');
            const v = $('val-output'); if (v) { v.textContent = Utils.fmt(state.output_gate.mean, 3); v.classList.add('visible'); }
            break;
        }
        case 7: {
            $('pipe-cs-tanh')?.classList.add('active-cell');
            const oT = $('op-tanh-cs'); if (oT) { oT.style.stroke = 'var(--candidate)'; oT.style.filter = 'drop-shadow(0 0 10px var(--candidate))'; }
            $('pipe-tanh-to-outmul')?.classList.add('active-cell');
            const oM = $('op-output-mul'); if (oM) { oM.style.stroke = 'var(--output-gate)'; oM.style.filter = 'drop-shadow(0 0 12px var(--output-gate))'; }
            $('pipe-out-hidden')?.classList.add('active-hidden');
            $('hs-lane')?.classList.add('active');
            $('cs-lane')?.classList.add('active');
            $('cs-lane')?.classList.add('pulse');
            updateBattery(state.cell_state.mean, 'final');
            updateStateOutputValues(state.cell_state.mean, state.hidden_state.mean);
            updateConnectors();
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
        forget: { icon: '🟠', name: 'Forget Gate (Cổng Quên)', desc: 'Quyết định giữ lại bao nhiêu thông tin cũ từ Cell State.\nGiá trị gần 1 → giữ nhiều, gần 0 → quên nhiều.', formula: state.formulas.forget, value: state.forget_gate.mean, interpret: v => `→ Giữ lại ${(v * 100).toFixed(0)}% thông tin cũ` },
        input: { icon: '🟢', name: 'Input Gate (Cổng Nhập)', desc: 'Quyết định thông tin mới nào sẽ được lưu vào Cell State.', formula: state.formulas.input, value: state.input_gate.mean, interpret: v => `→ Cho ${(v * 100).toFixed(0)}% thông tin mới vào` },
        candidate: { icon: '🔵', name: 'Candidate (Giá Trị Ứng Viên)', desc: 'Tạo vector thông tin mới có thể thêm vào Cell State.\nDùng hàm tanh (giá trị từ -1 đến 1).', formula: state.formulas.candidate, value: state.candidate.mean, interpret: v => `→ Giá trị trung bình: ${Utils.fmt(v, 3)}` },
        output: { icon: '🟣', name: 'Output Gate (Cổng Xuất)', desc: 'Quyết định phần nào của Cell State sẽ xuất ra thành Hidden State.', formula: state.formulas.output, value: state.output_gate.mean, interpret: v => `→ Xuất ra ${(v * 100).toFixed(0)}% thông tin` }
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

    // ★ Show section immediately and scroll to it
    section.classList.add('active');
    section.scrollIntoView({ behavior: 'smooth', block: 'center' });
    await Utils.sleep(600);

    // ★ Animate positive bar with count-up
    const barPos = document.getElementById('bar-positive');
    const pctPos = document.getElementById('pct-positive');
    const posPct = pred.probabilities.positive * 100;
    await animateBar(barPos, pctPos, posPct, 800);
    await Utils.sleep(300);

    // ★ Animate negative bar with count-up
    const barNeg = document.getElementById('bar-negative');
    const pctNeg = document.getElementById('pct-negative');
    const negPct = pred.probabilities.negative * 100;
    await animateBar(barNeg, pctNeg, negPct, 800);
    await Utils.sleep(500);

    // ★ Dramatic conclusion reveal
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

    // ★ Glow pulse on conclusion
    await Utils.sleep(600);
    conclusion.style.boxShadow = pred.label === 'Tích cực'
        ? '0 0 30px rgba(34, 197, 94, 0.5), 0 0 60px rgba(34, 197, 94, 0.2)'
        : '0 0 30px rgba(239, 68, 68, 0.5), 0 0 60px rgba(239, 68, 68, 0.2)';
    await Utils.sleep(1000);
    conclusion.style.boxShadow = '';

    anim.stopAutoPlay();
    document.getElementById('btn-autoplay').textContent = '▶▶ Tự Động';
}

// ★ Smooth bar + count-up animation
function animateBar(barEl, pctEl, targetPct, duration) {
    return new Promise(resolve => {
        const t0 = performance.now();
        function tick(now) {
            const p = Math.min((now - t0) / duration, 1);
            const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
            const current = targetPct * eased;
            barEl.style.width = current + '%';
            pctEl.textContent = current.toFixed(1) + '%';
            if (p < 1) {
                requestAnimationFrame(tick);
            } else {
                pctEl.textContent = targetPct.toFixed(1) + '%';
                resolve();
            }
        }
        requestAnimationFrame(tick);
    });
}
