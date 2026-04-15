/**
 * Utility functions
 */
const Utils = {
    /** Delay promise */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },

    /** Format number to fixed decimals */
    fmt(num, decimals = 4) {
        if (typeof num !== 'number') return '0';
        return num.toFixed(decimals);
    },

    /** Summarize vector for display */
    summarizeVector(vec, n = 3) {
        if (!vec || vec.length === 0) return '[]';
        const shown = vec.slice(0, n).map(v => Utils.fmt(v, 2));
        return `[${shown.join(', ')}${vec.length > n ? ', ···' : ''}]`;
    },

    /** Map value to color intensity (0-1 -> transparent to opaque) */
    valueToOpacity(value, min = 0, max = 1) {
        return Math.max(0.1, Math.min(1, (value - min) / (max - min)));
    },

    /** Create SVG element */
    svgEl(tag, attrs = {}) {
        const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [k, v] of Object.entries(attrs)) {
            el.setAttribute(k, v);
        }
        return el;
    },

    /** Create HTML element */
    htmlEl(tag, className = '', innerHTML = '') {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (innerHTML) el.innerHTML = innerHTML;
        return el;
    },

    /** Smooth scroll to element */
    scrollTo(el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    },

    /** Clamp value between min and max */
    clamp(val, min, max) {
        return Math.max(min, Math.min(max, val));
    },

    /** Percentage string */
    pct(val) {
        return (val * 100).toFixed(1) + '%';
    }
};
