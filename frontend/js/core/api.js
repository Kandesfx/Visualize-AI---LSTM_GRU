/**
 * API Client - Giao tiếp với backend FastAPI
 */
const API = {
    BASE_URL: '',

    async predictLSTM(text) {
        return this._post('/api/predict/lstm', { text });
    },

    async predictGRU(text) {
        return this._post('/api/predict/gru', { text });
    },

    async healthCheck() {
        return this._get('/api/health');
    },

    async _post(endpoint, data) {
        try {
            const res = await fetch(this.BASE_URL + endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Lỗi server');
            }
            return await res.json();
        } catch (e) {
            console.error('API Error:', e);
            throw e;
        }
    },

    async _get(endpoint) {
        try {
            const res = await fetch(this.BASE_URL + endpoint);
            if (!res.ok) throw new Error('Lỗi server');
            return await res.json();
        } catch (e) {
            console.error('API Error:', e);
            throw e;
        }
    }
};
