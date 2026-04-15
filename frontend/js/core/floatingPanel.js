/**
 * FloatingPanel — Draggable, snap-to-edge, collapsible controls panel
 * Used on both LSTM and GRU demo pages
 */
class FloatingPanel {
    constructor(panelId) {
        this.panel = document.getElementById(panelId);
        this.handle = this.panel.querySelector('.panel-handle');
        this.toggleBtn = this.panel.querySelector('.panel-toggle');
        this.body = this.panel.querySelector('.panel-body');

        this.isDragging = false;
        this.collapsed = false;
        this.startX = 0;
        this.startY = 0;
        this.offsetX = 0;
        this.offsetY = 0;

        // Snap threshold (px from edge)
        this.snapThreshold = 60;

        this._initPosition();
        this._bindEvents();
    }

    _initPosition() {
        // Start snapped to bottom-right, collapsed
        this.panel.style.position = 'fixed';
        this.panel.style.bottom = '20px';
        this.panel.style.right = '20px';
        this.panel.style.top = 'auto';
        this.panel.style.left = 'auto';
        this.panel.style.zIndex = '2000';
        // Start expanded so user can see controls
        this.collapsed = false;
    }

    _bindEvents() {
        // Drag start
        this.handle.addEventListener('mousedown', (e) => this._onDragStart(e));
        this.handle.addEventListener('touchstart', (e) => this._onDragStart(e), { passive: false });

        // Drag move/end on document
        document.addEventListener('mousemove', (e) => this._onDragMove(e));
        document.addEventListener('mouseup', () => this._onDragEnd());
        document.addEventListener('touchmove', (e) => this._onDragMove(e), { passive: false });
        document.addEventListener('touchend', () => this._onDragEnd());
    }

    _getPos(e) {
        if (e.touches) return { x: e.touches[0].clientX, y: e.touches[0].clientY };
        return { x: e.clientX, y: e.clientY };
    }

    _onDragStart(e) {
        e.preventDefault();
        this.isDragging = true;
        this.panel.classList.add('dragging');

        const pos = this._getPos(e);
        const rect = this.panel.getBoundingClientRect();
        this.offsetX = pos.x - rect.left;
        this.offsetY = pos.y - rect.top;

        // Switch from bottom/right to top/left positioning for drag
        this.panel.style.left = rect.left + 'px';
        this.panel.style.top = rect.top + 'px';
        this.panel.style.right = 'auto';
        this.panel.style.bottom = 'auto';
    }

    _onDragMove(e) {
        if (!this.isDragging) return;
        e.preventDefault();

        const pos = this._getPos(e);
        let newX = pos.x - this.offsetX;
        let newY = pos.y - this.offsetY;

        // Clamp to viewport
        const rect = this.panel.getBoundingClientRect();
        newX = Math.max(0, Math.min(window.innerWidth - rect.width, newX));
        newY = Math.max(0, Math.min(window.innerHeight - rect.height, newY));

        this.panel.style.left = newX + 'px';
        this.panel.style.top = newY + 'px';
    }

    _onDragEnd() {
        if (!this.isDragging) return;
        this.isDragging = false;
        this.panel.classList.remove('dragging');

        // Check snap-to-edge
        const rect = this.panel.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;

        // Snap to nearest horizontal edge
        if (rect.left < this.snapThreshold) {
            // Snap left
            this.panel.style.left = '8px';
            this.panel.classList.add('snapped-left');
            this.panel.classList.remove('snapped-right');
        } else if (rect.right > vw - this.snapThreshold) {
            // Snap right
            this.panel.style.left = (vw - rect.width - 8) + 'px';
            this.panel.classList.add('snapped-right');
            this.panel.classList.remove('snapped-left');
        } else {
            this.panel.classList.remove('snapped-left', 'snapped-right');
        }

        // Snap to nearest vertical edge
        if (rect.top < this.snapThreshold) {
            this.panel.style.top = '8px';
        } else if (rect.bottom > vh - this.snapThreshold) {
            this.panel.style.top = (vh - rect.height - 8) + 'px';
        }
    }

    toggle() {
        this.collapsed = !this.collapsed;
        if (this.collapsed) {
            this.panel.classList.add('collapsed');
            this.toggleBtn.textContent = '▷';
            this.toggleBtn.title = 'Mở rộng panel';
        } else {
            this.panel.classList.remove('collapsed');
            this.toggleBtn.textContent = '◁';
            this.toggleBtn.title = 'Thu gọn panel';
        }
    }

    show() {
        this.panel.style.display = '';
        this._initPosition();
    }

    hide() {
        this.panel.style.display = 'none';
    }
}
