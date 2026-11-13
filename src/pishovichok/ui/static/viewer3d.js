// ESM Vue 3 компонент для 3D-визуализации молекул на базе локального 3Dmol.js.
// Поддерживает единый WebGL-контекст ($3Dmol.createViewer) без пересоздания DOM.

function ensure3DmolLoaded() {
  if (window.$3Dmol) {
    return Promise.resolve(window.$3Dmol);
  }
  if (!window.__pishovichok3DmolPromise) {
    window.__pishovichok3DmolPromise = new Promise((resolve, reject) => {
      const existing = document.querySelector('script[src="/static/3Dmol-min.js"]');
      if (existing) {
        existing.addEventListener('load', () => resolve(window.$3Dmol));
        existing.addEventListener('error', reject);
        if (window.$3Dmol) resolve(window.$3Dmol);
        return;
      }
      const script = document.createElement('script');
      script.src = '/static/3Dmol-min.js';
      script.async = true;
      script.onload = () => resolve(window.$3Dmol);
      script.onerror = (err) => reject(err);
      document.head.appendChild(script);
    });
  }
  return window.__pishovichok3DmolPromise;
}

export default {
  template: `
    <div style="position:relative; width:100%; height:100%; min-height:260px; border-radius:8px; overflow:hidden; border:1px solid #E2E8F0; background:#F8FAFC;">
      <div ref="container" style="width:100%; height:100%; min-height:260px;"></div>
      <div v-if="empty" style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#475569; font-size:12px; pointer-events:none; text-align:center; padding:12px;">
        Генерация 3D-конформера (ETKDGv3)...
      </div>
    </div>
  `,
  props: {
    backgroundColor: { type: String, default: '#F8FAFC' },
  },
  data() {
    return {
      empty: true,
    };
  },
  async mounted() {
    await ensure3DmolLoaded();
    if (!this.$refs.container || !window.$3Dmol) return;

    this._viewer = window.$3Dmol.createViewer(this.$refs.container, {
      backgroundColor: this.backgroundColor,
    });

    if (typeof ResizeObserver !== 'undefined') {
      this._resizeObserver = new ResizeObserver(() => {
        if (this._viewer) {
          this._viewer.resize();
          this._viewer.render();
        }
      });
      this._resizeObserver.observe(this.$refs.container);
    }
  },
  beforeUnmount() {
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
    if (this._viewer) {
      this._viewer.clear();
      this._viewer = null;
    }
  },
  methods: {
    async loadMolBlock(molblock, highlightIndices = []) {
      await ensure3DmolLoaded();
      if (!this._viewer && this.$refs.container && window.$3Dmol) {
        this._viewer = window.$3Dmol.createViewer(this.$refs.container, {
          backgroundColor: this.backgroundColor,
        });
      }
      if (!this._viewer) return;

      this._viewer.clear();
      if (!molblock) {
        this.empty = true;
        this._viewer.render();
        return;
      }

      this.empty = false;
      this._viewer.addModel(molblock, 'sdf');
      this._viewer.setStyle(
        {},
        {
          stick: { radius: 0.14, colorscheme: 'Jmol' },
          sphere: { scale: 0.22, colorscheme: 'Jmol' },
        },
      );

      if (Array.isArray(highlightIndices) && highlightIndices.length > 0) {
        this._viewer.setStyle(
          { index: highlightIndices },
          {
            stick: { radius: 0.22, color: '#22c55e' },
            sphere: { scale: 0.30, color: '#22c55e' },
          },
        );
      }

      this._viewer.zoomTo();
      this._viewer.render();
    },
    async loadDockingComplex(ligandSdf, receptorPdb = '', keyResidues = []) {
      await ensure3DmolLoaded();
      if (!this._viewer && this.$refs.container && window.$3Dmol) {
        this._viewer = window.$3Dmol.createViewer(this.$refs.container, {
          backgroundColor: this.backgroundColor,
        });
      }
      if (!this._viewer) return;

      this._viewer.clear();
      if (!ligandSdf && !receptorPdb) {
        this.empty = true;
        this._viewer.render();
        return;
      }

      this.empty = false;
      if (receptorPdb) {
        this._viewer.addModel(receptorPdb, 'pdb');
        this._viewer.setStyle(
          { model: 0 },
          {
            cartoon: { color: '#CBD5E1', opacity: 0.72 },
          },
        );
        if (Array.isArray(keyResidues) && keyResidues.length > 0) {
          const resNums = keyResidues
            .map((lbl) => parseInt(String(lbl).replace(/^\D+/, ''), 10))
            .filter((n) => !Number.isNaN(n));
          if (resNums.length > 0) {
            this._viewer.setStyle(
              { model: 0, resi: resNums },
              {
                cartoon: { color: '#CBD5E1', opacity: 0.65 },
                stick: { radius: 0.12, color: '#64748B' },
              },
            );
          }
        }
      }

      if (ligandSdf) {
        const ligModelIdx = receptorPdb ? 1 : 0;
        this._viewer.addModel(ligandSdf, 'sdf');
        this._viewer.setStyle(
          { model: ligModelIdx },
          {
            stick: { radius: 0.20, colorscheme: 'Jmol' },
            sphere: { scale: 0.24, colorscheme: 'Jmol' },
          },
        );
        this._viewer.zoomTo({ model: ligModelIdx });
      } else {
        this._viewer.zoomTo();
      }
      this._viewer.render();
    },
    clear() {
      this.clearViewer();
    },
    clearViewer() {
      if (this._viewer) {
        this._viewer.clear();
        this._viewer.render();
      }
      this.empty = true;
    },
  },
};
