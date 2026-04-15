/**
 * Animation Engine - GSAP Timeline Controller
 * Quản lý tất cả animations cho demo page
 */
class AnimationEngine {
    constructor() {
        this.speed = 1;
        this.isPlaying = false;
        this.currentStep = 0;
        this.totalSteps = 0;
        this.onStepChange = null;
        this.onComplete = null;
        this._autoPlayTimer = null;
    }

    setSpeed(speed) {
        this.speed = speed;
    }

    getStepDuration() {
        return 1500 / this.speed;
    }

    getPauseDuration() {
        return 300 / this.speed;
    }

    /** Animate element with CSS transitions */
    async animate(element, properties, duration = 500) {
        return new Promise(resolve => {
            const dur = duration / this.speed;
            element.style.transition = `all ${dur}ms ease`;
            Object.assign(element.style, properties);
            setTimeout(resolve, dur);
        });
    }

    /** Fade in element */
    async fadeIn(element, duration = 400) {
        element.style.opacity = '0';
        element.style.display = '';
        await Utils.sleep(10);
        return this.animate(element, { opacity: '1' }, duration);
    }

    /** Fade out element */
    async fadeOut(element, duration = 400) {
        await this.animate(element, { opacity: '0' }, duration);
        element.style.display = 'none';
    }

    /** Scale in element */
    async scaleIn(element, duration = 400) {
        element.style.opacity = '0';
        element.style.transform = 'scale(0.5)';
        element.style.display = '';
        await Utils.sleep(10);
        return this.animate(element, { opacity: '1', transform: 'scale(1)' }, duration);
    }

    /** Stagger animate children */
    async stagger(elements, properties, staggerDelay = 100, duration = 400) {
        const promises = [];
        for (let i = 0; i < elements.length; i++) {
            const delay = (staggerDelay * i) / this.speed;
            promises.push(
                Utils.sleep(delay).then(() => this.animate(elements[i], properties, duration))
            );
        }
        return Promise.all(promises);
    }

    /** Animate SVG path drawing */
    async drawPath(pathElement, duration = 800) {
        const length = pathElement.getTotalLength();
        pathElement.style.strokeDasharray = length;
        pathElement.style.strokeDashoffset = length;
        pathElement.style.transition = `stroke-dashoffset ${duration / this.speed}ms ease`;
        await Utils.sleep(10);
        pathElement.style.strokeDashoffset = '0';
        return Utils.sleep(duration / this.speed);
    }

    /** Animate particle along SVG path */
    async moveParticle(particle, pathElement, duration = 600) {
        const dur = duration / this.speed;
        const length = pathElement.getTotalLength();
        const startTime = performance.now();

        particle.style.opacity = '1';

        return new Promise(resolve => {
            const step = (timestamp) => {
                const elapsed = timestamp - startTime;
                const progress = Math.min(elapsed / dur, 1);
                const point = pathElement.getPointAtLength(progress * length);
                particle.setAttribute('cx', point.x);
                particle.setAttribute('cy', point.y);

                if (progress < 1) {
                    requestAnimationFrame(step);
                } else {
                    particle.style.opacity = '0';
                    resolve();
                }
            };
            requestAnimationFrame(step);
        });
    }

    /** Pulse animation on element */
    async pulse(element, color, duration = 600) {
        const dur = duration / this.speed;
        const origFilter = element.style.filter || '';
        element.style.transition = `filter ${dur / 2}ms ease`;
        element.style.filter = `drop-shadow(0 0 12px ${color}) brightness(1.3)`;
        await Utils.sleep(dur / 2);
        element.style.filter = origFilter;
        return Utils.sleep(dur / 2);
    }

    /** Start auto play */
    startAutoPlay(stepFunction) {
        this.isPlaying = true;
        const run = async () => {
            if (!this.isPlaying) return;
            await stepFunction();
            if (this.isPlaying && this.currentStep < this.totalSteps) {
                this._autoPlayTimer = setTimeout(run, this.getPauseDuration());
            } else {
                this.isPlaying = false;
            }
        };
        run();
    }

    /** Stop auto play */
    stopAutoPlay() {
        this.isPlaying = false;
        if (this._autoPlayTimer) {
            clearTimeout(this._autoPlayTimer);
            this._autoPlayTimer = null;
        }
    }
}
