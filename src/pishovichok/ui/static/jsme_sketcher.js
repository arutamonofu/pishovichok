// ESM Vue 3 компонент 2D-скетчера молекул на базе локального дистрибутива JSME.
// Перехватывает глобальный коллбэк window.jsmeOnLoad и эмиттит событие smiles-changed.

function ensureJsmeLoaded() {
  if (window.JSApplet && window.JSApplet.JSME) {
    return Promise.resolve(window.JSApplet);
  }
  if (!window.__pishovichokJsmePromise) {
    window.__pishovichokJsmePromise = new Promise((resolve, reject) => {
      const prevOnLoad = window.jsmeOnLoad;
      window.jsmeOnLoad = () => {
        if (typeof prevOnLoad === 'function') {
          try {
            prevOnLoad();
          } catch (_e) {
            // ignore chained callback errors
          }
        }
        resolve(window.JSApplet);
      };

      const existing = document.querySelector('script[src="/static/jsme/jsme.nocache.js"]');
      if (existing) {
        if (window.JSApplet && window.JSApplet.JSME) {
          resolve(window.JSApplet);
        }
        return;
      }
      const script = document.createElement('script');
      script.src = '/static/jsme/jsme.nocache.js';
      script.async = true;
      script.onerror = (err) => reject(err);
      document.head.appendChild(script);
    });
  }
  return window.__pishovichokJsmePromise;
}

let jsmeInstanceCounter = 0;

export default {
  template: `
    <div ref="wrapper" style="width:100%; height:240px; min-height:200px; border-radius:8px; overflow:hidden; border:1px solid rgba(148,163,184,0.25); background:#ffffff;">
      <div :id="containerId" ref="jsmeContainer" style="width:100%; height:100%;"></div>
    </div>
  `,
  props: {
    initialSmiles: { type: String, default: '' },
    options: { type: String, default: 'query,hydrogens' },
  },
  data() {
    jsmeInstanceCounter += 1;
    return {
      containerId: `pishovichok_jsme_${jsmeInstanceCounter}_${Date.now()}`,
    };
  },
  async mounted() {
    await ensureJsmeLoaded();
    if (!this.$refs.wrapper || !window.JSApplet || !window.JSApplet.JSME) return;

    const rect = this.$refs.wrapper.getBoundingClientRect();
    const w = `${Math.max(240, Math.floor(rect.width || 300))}px`;
    const h = `${Math.max(200, Math.floor(rect.height || 240))}px`;

    this._suppressEmit = false;
    this._jsme = new window.JSApplet.JSME(this.containerId, w, h, {
      options: this.options,
    });

    this._jsme.setCallBack('AfterStructureModified', () => {
      if (this._suppressEmit || !this._jsme) return;
      const smiles = this._jsme.smiles() || '';
      this.$emit('smiles-changed', { smiles });
    });

    if (this.initialSmiles) {
      this.setSmiles(this.initialSmiles);
    }

    if (typeof ResizeObserver !== 'undefined') {
      this._resizeObserver = new ResizeObserver((entries) => {
        if (!this._jsme || !entries.length) return;
        const { width, height } = entries[0].contentRect;
        if (width > 120 && height > 120) {
          this._jsme.setSize(`${Math.floor(width)}px`, `${Math.floor(height)}px`);
        }
      });
      this._resizeObserver.observe(this.$refs.wrapper);
    }
  },
  beforeUnmount() {
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
    this._jsme = null;
  },
  methods: {
    setSmiles(smiles) {
      if (!this._jsme) return;
      this._suppressEmit = true;
      try {
        this._jsme.readGenericMolecularInput(smiles || '');
      } finally {
        setTimeout(() => {
          this._suppressEmit = false;
        }, 50);
      }
    },
  },
};
