// Опциональный адаптер для статического бандла ketcher-standalone (/static/ketcher/index.html).
// Активируется при настройке PISHOVICHOK_SKETCHER_BACKEND=ketcher и наличии локальных ассетов Ketcher.

export default {
  template: `
    <div style="width:100%; height:240px; min-height:200px; border-radius:8px; overflow:hidden; border:1px solid rgba(148,163,184,0.25); background:#ffffff;">
      <iframe
        ref="frame"
        src="/static/ketcher/index.html"
        style="width:100%; height:100%; border:none;"
        @load="onFrameLoad"
      ></iframe>
    </div>
  `,
  props: {
    initialSmiles: { type: String, default: '' },
  },
  methods: {
    onFrameLoad() {
      const ketcher = this.$refs.frame?.contentWindow?.ketcher;
      if (!ketcher) return;
      this._ketcher = ketcher;
      if (this.initialSmiles) {
        this.setSmiles(this.initialSmiles);
      }
      if (ketcher.editor && ketcher.editor.subscribe) {
        ketcher.editor.subscribe('change', async () => {
          try {
            const smiles = await ketcher.getSmiles();
            this.$emit('smiles-changed', { smiles: smiles || '' });
          } catch (_e) {
            // ignore transient empty canvas errors
          }
        });
      }
    },
    async setSmiles(smiles) {
      if (this._ketcher && typeof this._ketcher.setMolecule === 'function') {
        await this._ketcher.setMolecule(smiles || '');
      }
    },
  },
};
