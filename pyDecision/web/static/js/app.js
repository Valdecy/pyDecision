/* ============================================================
   pyDecision · MCDA Studio · client app
   ============================================================ */

(function () {
  'use strict';

  // ---------- state ----------
  const state = {
    catalog: null,        // {alphabetical, grouped, groups}
    methods: {},          // key -> method spec
    currentKey: null,
    currentMethod: null,
    listMode: 'alpha',
    searchQuery: '',
    lastResult: null,
    inputs: {},           // collected raw input arrays / params
    altLabels: [],        // user-named alt labels (defaults A1..An)
    critLabels: [],       // user-named criterion labels (defaults C1..Cn)
    prefillMode: 'empty',
  };

  // ---------- helpers ----------
  const $  = (sel, root=document) => root.querySelector(sel);
  const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, attrs={}, ...children) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k === 'on' && typeof v === 'object') {
        for (const [evt, fn] of Object.entries(v)) e.addEventListener(evt, fn);
      } else if (k === 'style' && typeof v === 'object') {
        Object.assign(e.style, v);
      } else if (k.startsWith('data-')) {
        e.setAttribute(k, v);
      } else if (k in e) e[k] = v;
      else e.setAttribute(k, v);
    }
    for (const c of children) {
      if (c == null) continue;
      e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return e;
  };
  const fmt = (v, digits=4) => {
    if (v == null || isNaN(v)) return '—';
    const n = Number(v);
    if (!isFinite(n)) return String(v);
    if (Math.abs(n) < 0.0001 && n !== 0) return n.toExponential(2);
    return n.toFixed(digits).replace(/\.?0+$/, '') || '0';
  };
  const toast = (msg, kind='') => {
    const t = $('#toast');
    t.textContent = msg;
    t.className = 'toast' + (kind ? ' is-' + kind : '');
    t.hidden = false;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { t.hidden = true; }, 4200);
  };

  // ---------- API ----------
  const API = {
    methods: () => fetch('/api/methods').then(r => r.json()),
    method:  (k) => fetch('/api/method/' + encodeURIComponent(k)).then(r => r.json()),
    run:     (payload) => fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    }).then(r => r.json()),
    example: (k) => fetch('/api/example/' + encodeURIComponent(k)).then(r => r.json()),
    health:  () => fetch('/api/health').then(r => r.json()).catch(() => null),
    shutdown:() => fetch('/api/shutdown', {method:'POST'}).then(r => r.json()).catch(() => null),
  };

  // ---------- bootstrap ----------
  async function boot() {
    try {
      state.catalog = await API.methods();
      state.catalog.alphabetical.forEach(m => state.methods[m.key] = m);
      $('#method-count').textContent = state.catalog.alphabetical.length + ' methods';
      renderMethodList();
      attachEvents();
      pollHealth();
      setInterval(pollHealth, 8000);
    } catch (err) {
      toast('Cannot reach the pyDecision server. Is it running?', 'error');
      console.error(err);
    }
  }

  async function pollHealth() {
    const h = await API.health();
    const pill = $('#status-pill');
    const txt = $('#status-text');
    if (!h) {
      pill.classList.add('is-error'); pill.classList.remove('is-busy');
      txt.textContent = 'Offline';
    } else {
      pill.classList.remove('is-error'); pill.classList.remove('is-busy');
      txt.textContent = 'Ready';
    }
  }

  // ---------- sidebar ----------
  function renderMethodList() {
    const list = $('#method-list');
    list.innerHTML = '';
    const q = state.searchQuery.trim().toLowerCase();

    const matches = (m) => {
      if (!q) return true;
      return [m.name, m.group, m.key, m.summary].some(s =>
        (s || '').toLowerCase().includes(q));
    };

    if (state.listMode === 'alpha') {
      const filtered = state.catalog.alphabetical.filter(matches);
      if (!filtered.length) {
        list.appendChild(el('div', {class:'list-empty'}, 'No methods match.'));
        return;
      }
      filtered.forEach(m => list.appendChild(renderMethodRow(m)));
    } else {
      const groups = state.catalog.groups;
      let total = 0;
      groups.forEach(g => {
        const ms = (state.catalog.grouped[g] || []).filter(matches);
        if (!ms.length) return;
        list.appendChild(el('div', {class:'method-group-header'}, g));
        ms.forEach(m => list.appendChild(renderMethodRow(m)));
        total += ms.length;
      });
      if (!total) list.appendChild(el('div', {class:'list-empty'}, 'No methods match.'));
    }
  }

  function renderMethodRow(m) {
    const row = el('div', {
      class: 'method-row' + (state.currentKey === m.key ? ' is-active' : ''),
      'data-key': m.key,
      role: 'option',
      on: { click: () => selectMethod(m.key) },
    },
      el('span', {class:'method-row-mark'}),
      el('div', {class:'method-row-info'},
        el('div', {class:'method-row-name'}, m.name),
        el('div', {class:'method-row-group'}, m.group)
      )
    );
    return row;
  }

  // ---------- method selection ----------
  async function selectMethod(key) {
    state.currentKey = key;
    state.lastResult = null;
    state.prefillMode = 'empty';
    $$('.method-row').forEach(r => r.classList.toggle('is-active', r.dataset.key === key));
    try {
      state.currentMethod = await API.method(key);
    } catch (err) {
      toast('Failed to load method: ' + key, 'error');
      return;
    }
    renderMethodHeader(state.currentMethod);
    $('#tabs').hidden = false;
    activateTab('input');
    state.prefillMode = 'empty';
    renderInputForm(state.currentMethod);
    $('#empty-results').hidden = false;
    $('#results-content').hidden = true;
    $('#results-content').innerHTML = '';
  }

  function renderMethodHeader(m) {
    const head = $('#method-header');
    const badges = [
      el('span', {class:'badge badge-amber'}, m.group),
      el('span', {class:'badge'}, m.shape),
    ];
    if (m.weighting) badges.push(el('span', {class:'badge badge-cyan'}, 'computes weights'));
    if (m.ignores_weights) badges.push(el('span', {class:'badge'}, 'no weights needed'));
    head.innerHTML = '';
    head.appendChild(el('div', {class:'method-header-content'},
      el('div', {class:'method-header-text'},
        el('div', {class:'method-header-eyebrow'}, m.group),
        el('h1', {class:'method-header-title'}, m.name),
        el('p', {class:'method-header-summary'}, m.summary || '')
      ),
      el('div', {class:'method-header-badges'}, ...badges)
    ));
  }

  // ---------- tabs ----------
  function activateTab(name) {
    $$('.tab-btn').forEach(b => b.classList.toggle('is-active', b.dataset.tab === name));
    $('#panel-input').hidden = name !== 'input';
    $('#panel-results').hidden = name !== 'results';
  }

  // ---------- INPUT FORM ===============================================
  function renderInputForm(m) {
    const fields = $('#input-fields');
    fields.innerHTML = '';
    const dimC = $('#dim-controls');
    dimC.hidden = true;
    $('#dim-classes-control').hidden = true;
    $('#dim-experts-control').hidden = true;

    // Each shape gets its own renderer
    const shape = m.shape;
    const r = SHAPE_RENDERERS[shape];
    if (!r) {
      fields.appendChild(el('div', {class:'console-block'},
        'No input renderer registered for shape: ' + shape));
      return;
    }
    r(m, fields);
  }

  // === Common builders ==================================================
  function buildSpreadsheetGrid({rows, cols, rowLabels, colLabels, key, defaultFn, fuzzy=false, pair=false}) {
    const wrap = el('div', {class:'grid-wrap'});
    const table = el('table', {class:'grid'});
    const thead = el('thead');
    const headRow = el('tr');
    headRow.appendChild(el('th', {class:'row-head'}, ''));
    for (let j = 0; j < cols; j++) {
      headRow.appendChild(el('th', {}, colLabels ? (colLabels[j] || ('C'+(j+1))) : ('C'+(j+1))));
    }
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = el('tbody');
    tbody.dataset.rows = rows;
    tbody.dataset.cols = cols;
    tbody.dataset.fuzzy = fuzzy ? '1' : '0';
    tbody.dataset.pair = pair ? '1' : '0';
    if (key) tbody.dataset.key = key;

    for (let i = 0; i < rows; i++) {
      const tr = el('tr');
      tr.appendChild(el('th', {}, rowLabels ? (rowLabels[i] || ('A'+(i+1))) : ('A'+(i+1))));
      for (let j = 0; j < cols; j++) {
        const td = el('td');
        if (fuzzy) {
          const wrap3 = el('div', {class:'fuzzy-cell'});
          for (let k = 0; k < 3; k++) {
            const inp = el('input', {type:'text', value: defaultFn ? defaultFn(i, j, k) : '0'});
            inp.dataset.r = i; inp.dataset.c = j; inp.dataset.k = k;
            wrap3.appendChild(inp);
          }
          td.appendChild(wrap3);
        } else if (pair) {
          const wrap2 = el('div', {class:'fuzzy-cell'});
          for (let k = 0; k < 2; k++) {
            const inp = el('input', {type:'text', value: defaultFn ? defaultFn(i, j, k) : '0'});
            inp.dataset.r = i; inp.dataset.c = j; inp.dataset.k = k;
            wrap2.appendChild(inp);
          }
          td.appendChild(wrap2);
        } else {
          const inp = el('input', {type:'text', value: defaultFn ? defaultFn(i, j) : ''});
          inp.dataset.r = i; inp.dataset.c = j;
          td.appendChild(inp);
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  // Read a non-fuzzy grid as 2D array; fuzzy as 3D
  function readGrid(tbody) {
    const rows = parseInt(tbody.dataset.rows);
    const cols = parseInt(tbody.dataset.cols);
    const fuzzy = tbody.dataset.fuzzy === '1';
    const pair = tbody.dataset.pair === '1';
    const out = [];
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) {
        if (fuzzy) {
          const triple = [];
          for (let k = 0; k < 3; k++) {
            const inp = tbody.querySelector(`input[data-r="${i}"][data-c="${j}"][data-k="${k}"]`);
            triple.push(parseFloat(inp.value) || 0);
          }
          row.push(triple);
        } else if (pair) {
          const pairVals = [];
          for (let k = 0; k < 2; k++) {
            const inp = tbody.querySelector(`input[data-r="${i}"][data-c="${j}"][data-k="${k}"]`);
            pairVals.push(parseFloat(inp.value) || 0);
          }
          row.push(pairVals);
        } else {
          const inp = tbody.querySelector(`input[data-r="${i}"][data-c="${j}"]`);
          const v = inp.value.trim();
          row.push(v === '' ? 0 : (parseFloat(v) || 0));
        }
      }
      out.push(row);
    }
    return out;
  }

  // ---- vector input ----
  function buildVector({len, label, hint, valueFn, fuzzy=false, idPrefix='v'}) {
    const wrap = el('div', {class: 'input-section'});
    wrap.appendChild(buildSectionHead(label, hint));
    const grid = el('div', {class:'grid-wrap'});
    const table = el('table', {class:'grid'});
    const trh = el('tr');
    trh.appendChild(el('th', {class:'row-head'}, ''));
    for (let i = 0; i < len; i++) trh.appendChild(el('th', {}, 'C'+(i+1)));
    table.appendChild(el('thead', {}, trh));
    const tbody = el('tbody');
    tbody.dataset.rows = 1; tbody.dataset.cols = len;
    tbody.dataset.fuzzy = fuzzy ? '1' : '0';
    tbody.dataset.key = idPrefix;
    const tr = el('tr');
    tr.appendChild(el('th', {}, label.split(' ')[0]));
    for (let j = 0; j < len; j++) {
      const td = el('td');
      if (fuzzy) {
        const w3 = el('div', {class:'fuzzy-cell'});
        for (let k = 0; k < 3; k++) {
          const inp = el('input', {type:'text', value: valueFn ? valueFn(j, k) : '0'});
          inp.dataset.r = 0; inp.dataset.c = j; inp.dataset.k = k;
          w3.appendChild(inp);
        }
        td.appendChild(w3);
      } else {
        const inp = el('input', {type:'text', value: valueFn ? valueFn(j) : ''});
        inp.dataset.r = 0; inp.dataset.c = j;
        td.appendChild(inp);
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    table.appendChild(tbody);
    grid.appendChild(table);
    wrap.appendChild(grid);
    return wrap;
  }


  function buildTextAreaSection({label, hint, key, value=''}) {
    const wrap = el('div', {class:'input-section'});
    wrap.appendChild(buildSectionHead(label, hint));
    const ta = el('textarea', {
      class:'code-textarea',
      value: value || '',
      rows: 8,
      'data-key': key,
      spellcheck: false,
      placeholder: 'Paste a Python/JSON-like list or dict here'
    });
    wrap.appendChild(ta);
    return wrap;
  }

  function parseLooseStructuredText(raw) {
    const txt = String(raw || '').trim();
    if (!txt) return null;
    try { return JSON.parse(txt); } catch (_) {}
    let norm = txt
      .replace(/True/g, 'true')
      .replace(/False/g, 'false')
      .replace(/None/g, 'null')
      .replace(/'/g, '"')
      .replace(/\(/g, '[')
      .replace(/\)/g, ']');
    try { return JSON.parse(norm); } catch (_) {}
    throw new Error('Could not parse structured text for ' + txt.slice(0, 40));
  }

  // Pretty-print structured values for textareas. Specialises for the
  // "dict of profiles" shape CPP-Tri uses:
  //   { "Profile 1": [[5,5,5,5,5,5,5]],
  //     "Profile 2": [[20,20,...],[30,30,...]], ... }
  // Each profile gets one line, with row arrays inlined and numeric columns
  // padded so they line up. Falls back to JSON.stringify for other shapes.
  function formatStructured(v) {
    if (Array.isArray(v)) return JSON.stringify(v);
    if (v && typeof v === 'object') {
      const entries = Object.entries(v);
      const isProfileDict = entries.length > 0 && entries.every(([, val]) =>
        Array.isArray(val) && val.length > 0 && val.every(row =>
          Array.isArray(row) && row.every(x => typeof x === 'number')));
      if (!isProfileDict) return JSON.stringify(v, null, 2);
      // Compute per-column widths across every row
      let nCols = 0;
      entries.forEach(([, rows]) => rows.forEach(r => {
        if (r.length > nCols) nCols = r.length;
      }));
      const widths = new Array(nCols).fill(1);
      entries.forEach(([, rows]) => rows.forEach(r => {
        r.forEach((x, j) => {
          const s = String(x);
          if (s.length > widths[j]) widths[j] = s.length;
        });
      }));
      const fmtRow = (row) => '[ ' + row.map((x, j) =>
        String(x).padStart(widths[j], ' ')).join(', ') + ' ]';
      const lines = ['{'];
      entries.forEach(([key, rows], idx) => {
        const inner = rows.map(fmtRow).join(' , ');
        const trail = idx === entries.length - 1 ? '' : ',';
        lines.push(`\t${JSON.stringify(key)}: [ ${inner} ]${trail}`);
      });
      lines.push('}');
      return lines.join('\n');
    }
    return String(v);
  }
  function readVector(wrap) {
    const tbody = wrap.querySelector('tbody');
    const data = readGrid(tbody);
    return tbody.dataset.fuzzy === '1' ? data[0] : data[0];
  }

  // criterion-type vector (max/min select)
  function buildCriterionTypeRow(n, defaults, opts={}) {
    const wrap = el('div', {class:'input-section'});
    const fixedValue = opts.fixedValue || null;
    const disabled = !!opts.disabled;
    wrap.appendChild(buildSectionHead('Criterion type', fixedValue ? `fixed to ${fixedValue}` : 'max = benefit, min = cost'));
    const grid = el('div', {class:'grid-wrap'});
    const table = el('table', {class:'grid'});
    const trh = el('tr');
    trh.appendChild(el('th', {class:'row-head'}, ''));
    for (let i = 0; i < n; i++) trh.appendChild(el('th', {}, 'C'+(i+1)));
    table.appendChild(el('thead', {}, trh));
    const tbody = el('tbody');
    tbody.dataset.rows = 1; tbody.dataset.cols = n; tbody.dataset.key = 'ctype';
    const tr = el('tr');
    tr.appendChild(el('th', {}, 'Type'));
    for (let j = 0; j < n; j++) {
      const td = el('td');
      const sel = el('select', {});
      if (fixedValue) {
        sel.appendChild(el('option', {value: fixedValue}, fixedValue));
      } else {
        sel.appendChild(el('option', {value:'max'}, 'max'));
        sel.appendChild(el('option', {value:'min'}, 'min'));
      }
      sel.value = fixedValue || (defaults && defaults[j] ? defaults[j] : 'max');
      sel.dataset.r = 0; sel.dataset.c = j;
      sel.disabled = disabled;
      td.appendChild(sel);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    table.appendChild(tbody);
    grid.appendChild(table);
    wrap.appendChild(grid);
    return wrap;
  }
  function readCriterionType(wrap) {
    return $$('select', wrap).map(s => s.value);
  }

  function buildSectionHead(title, hint) {
    return el('div', {class:'input-section-head'},
      el('span', {class:'input-section-title'}, title),
      hint ? el('span', {class:'input-section-hint'}, hint) : null
    );
  }

  function buildKVRow(items) {
    const row = el('div', {class:'kv-row'});
    items.forEach(item => {
      const kv = el('div', {class:'kv'},
        el('label', {}, item.label));
      let inp;
      if (item.type === 'select') {
        inp = el('select', {});
        (item.options || []).forEach(o => {
          // Accept three option shapes from the registry / shape renderers:
          //   "literal"               -> value = label = "literal"
          //   ["value", "label"]      -> tuple form (Python registry)
          //   {value, label}          -> object form
          let val, lbl;
          if (Array.isArray(o)) { val = String(o[0]); lbl = String(o[1] != null ? o[1] : o[0]); }
          else if (o && typeof o === 'object') { val = String(o.value); lbl = String(o.label != null ? o.label : o.value); }
          else { val = lbl = String(o); }
          const opt = el('option', {value: val}, lbl);
          inp.appendChild(opt);
        });
        if (item.value !== undefined && item.value !== null) {
          inp.value = String(item.value);
        }
      } else {
        inp = el('input', {type: item.type || 'text', value: item.value !== undefined ? String(item.value) : ''});
        if (item.placeholder) inp.placeholder = item.placeholder;
      }
      inp.dataset.key = item.key;
      kv.appendChild(inp);
      row.appendChild(kv);
    });
    return row;
  }

  // ===================== SHAPE RENDERERS =====================
  const DEMO = {
    matrix: [[999,8.5,24,9.2],[799,8.2,28,8.9],[899,8.0,26,9.0],[699,7.8,30,8.5],[599,7.5,22,8.0]],
    weights: [0.30, 0.25, 0.20, 0.25],
    ctype: ['min','max','max','max'],
  };

  const isDemoMode = () => state.prefillMode === 'demo';
  const demoScalar = (value) => isDemoMode() ? value : '';
  const demoWeightValue = (j, criteria) => isDemoMode()
    ? (DEMO.weights[j] !== undefined ? DEMO.weights[j] : (1/criteria).toFixed(3))
    : '';
  const demoProfileValue = (i, j, classes, includeEndPoints=false) => {
    if (!isDemoMode()) return '';
    const col = DEMO.matrix.map(r => r[j] || 0);
    const lo = Math.min(...col);
    const hi = Math.max(...col);
    if (includeEndPoints && classes <= 1) return lo.toFixed(2);
    if (includeEndPoints) {
      return (lo + (hi-lo) * i / Math.max(classes - 1, 1)).toFixed(2);
    }
    return (lo + (hi-lo) * (i+1) / Math.max(classes, 1)).toFixed(2);
  };

  function renderDimControls({alts=true, crits=true, classes=false, experts=false, defaults={}}) {
    const dimC = $('#dim-controls');
    dimC.hidden = false;
    $('#dim-alternatives-control').hidden = !alts;
    $('#dim-criteria-control').hidden = !crits;
    $('#dim-classes-control').hidden = !classes;
    $('#dim-experts-control').hidden = !experts;
    if (defaults.alternatives !== undefined) $('#num-alternatives').value = defaults.alternatives;
    if (defaults.criteria !== undefined) $('#num-criteria').value = defaults.criteria;
    if (classes && defaults.classes !== undefined) $('#num-classes').value = defaults.classes;
    if (experts && defaults.experts !== undefined) $('#num-experts').value = defaults.experts;
  }

  function getDims() {
    return {
      alternatives: parseInt($('#num-alternatives').value) || 2,
      criteria: parseInt($('#num-criteria').value) || 2,
      classes: parseInt($('#num-classes').value) || 3,
      experts: parseInt($('#num-experts').value) || 1,
    };
  }

  function defaultMatrixCell(i, j) {
    if (!isDemoMode()) return '';
    if (i < DEMO.matrix.length && j < DEMO.matrix[0].length) return DEMO.matrix[i][j];
    return '';
  }

  // === decision_matrix family ===
  function renderDM(m, container, opts={}) {
    const {alternatives, criteria} = getDims();
    renderDimControls({defaults: {alternatives, criteria}});

    // 1. matrix
    const matSec = el('div', {class:'input-section'});
    matSec.appendChild(buildSectionHead('Decision matrix',
      `${alternatives} alternatives × ${criteria} criteria — paste directly into the grid or use the paste button`));
    matSec.appendChild(buildSpreadsheetGrid({
      rows: alternatives, cols: criteria,
      key: 'matrix',
      defaultFn: defaultMatrixCell,
    }));
    container.appendChild(matSec);

    // 2. weights row (unless ignored)
    if (!opts.noWeights) {
      const wDefault = (i) => demoWeightValue(i, criteria);
      container.appendChild(buildVector({
        len: criteria, label: 'Weights',
        hint: 'must sum to 1', valueFn: wDefault, idPrefix: 'weights'
      }));
    }

    // 3. criterion types
    const fixedElectreMax = new Set(['electre_i','electre_i_s','electre_i_v','electre_ii','electre_iii','electre_iv']).has(m.key);
    container.appendChild(buildCriterionTypeRow(criteria, fixedElectreMax ? Array.from({length:criteria}, () => 'max') : (isDemoMode() ? DEMO.ctype : null), fixedElectreMax ? {fixedValue:'max', disabled:true} : {}));

    // 4. extra params
    if (m.params && m.params.length) {
      const ps = el('div', {class:'input-section'});
      ps.appendChild(buildSectionHead('Parameters', 'optional method-specific settings'));
      const items = m.params.map(p => paramAsKV(p));
      ps.appendChild(buildKVRow(items));
      container.appendChild(ps);
    }

    // 5. critical-criterion picker for ODO/OVO
    if (opts.critical) {
      const cs = el('div', {class:'input-section'});
      cs.appendChild(buildSectionHead('Critical criterion', 'index 0 = first criterion'));
      cs.appendChild(buildKVRow([
        {key:'critical_criterion', label:'Critical criterion index', type:'number', value:0}
      ]));
      container.appendChild(cs);
    }
  }

  function paramAsKV(p) {
    const item = {key: p.key, label: p.label || p.key};
    if (p.type === 'select') {
      item.type = 'select';
      item.options = p.options || [];
      // Default: explicit p.default; otherwise the first option's value
      // (handle both "literal" and ["value","label"] tuple shapes).
      let fallback = '';
      if (item.options.length) {
        const first = item.options[0];
        fallback = Array.isArray(first) ? first[0]
                 : (first && typeof first === 'object' ? first.value : first);
      }
      item.value = p.default !== undefined ? p.default : fallback;
    } else if (p.type === 'number' || p.type === 'float') {
      item.type = 'number';
      item.value = p.default !== undefined ? p.default : 0;
    } else {
      item.type = 'text';
      item.value = p.default !== undefined ? p.default : '';
    }
    return item;
  }

  const SHAPE_RENDERERS = {
    decision_matrix:           (m, c) => renderDM(m, c),
    decision_matrix_no_weights:(m, c) => renderDM(m, c, {noWeights: true}),
    decision_matrix_critical:  (m, c) => renderDM(m, c, {critical: true}),

    decision_matrix_utility: (m, c) => {
      renderDM(m, c);
      const {criteria} = getDims();
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Utility functions', 'one per criterion'));
      const items = [];
      for (let i = 0; i < criteria; i++) {
        items.push({key:`utility_${i}`, label:`C${i+1}`, type:'select',
          options: ['linear','exp','step','quadratic'],
          value:'linear'});
      }
      sec.appendChild(buildKVRow(items));
      c.appendChild(sec);
    },

    smart: (m, c) => {
      const {alternatives, criteria} = getDims();
      renderDimControls({defaults:{alternatives, criteria}});
      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Decision matrix', `${alternatives} × ${criteria}`));
      matSec.appendChild(buildSpreadsheetGrid({
        rows: alternatives, cols: criteria, key:'matrix', defaultFn: defaultMatrixCell
      }));
      c.appendChild(matSec);
      c.appendChild(buildVector({len:criteria, label:'Grades', hint:'1 = least important, K = most',
        valueFn:()=>3, idPrefix:'grades'}));
      c.appendChild(buildVector({len:criteria, label:'Lower bounds',
        valueFn:(j)=> isDemoMode() ? Math.min(...DEMO.matrix.map(r => r[j])) : '',
        idPrefix:'lower'}));
      c.appendChild(buildVector({len:criteria, label:'Upper bounds',
        valueFn:(j)=> isDemoMode() ? Math.max(...DEMO.matrix.map(r => r[j])) : '',
        idPrefix:'upper'}));
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));
    },

    weighting_only: (m, c) => {
      const {alternatives, criteria} = getDims();
      renderDimControls({defaults:{alternatives, criteria}});
      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Decision matrix',
        'weights are computed from data — no need to provide weights'));
      matSec.appendChild(buildSpreadsheetGrid({
        rows: alternatives, cols: criteria, key:'matrix', defaultFn: defaultMatrixCell
      }));
      c.appendChild(matSec);
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));
    },

    pairwise: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      // Hide the alternatives stepper since pairwise only uses criteria count
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Pairwise comparison matrix',
        'reciprocal-symmetric (Saaty 1–9 scale)'));
      const labels = Array.from({length: criteria}, (_, i) => 'C'+(i+1));
      const defaultPW = (i, j) => i === j ? 1 :
        (i < j ? Math.min(9, j-i+1) : (1/Math.min(9, i-j+1)).toFixed(3));
      sec.appendChild(buildSpreadsheetGrid({
        rows: criteria, cols: criteria,
        rowLabels: labels, colLabels: labels,
        key:'matrix', defaultFn: defaultPW,
      }));
      c.appendChild(sec);
      // auto reciprocal mirror on edit
      sec.querySelectorAll('input').forEach(inp => {
        inp.addEventListener('change', () => {
          const r = parseInt(inp.dataset.r), col = parseInt(inp.dataset.c);
          if (r === col) return;
          const v = parseFloat(inp.value);
          if (!isFinite(v) || v === 0) return;
          const mirror = sec.querySelector(`input[data-r="${col}"][data-c="${r}"]`);
          if (mirror) mirror.value = (1/v).toFixed(4);
        });
      });
    },

    ppf_pairwise: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('PPF pairwise matrix',
        'each cell is a proportional picture fuzzy pair (k1, k2)'));
      const labels = Array.from({length: criteria}, (_, i) => 'C'+(i+1));
      const defPPF = (i, j, k) => i === j ? 0 : (k === 0 ? Math.min(9, j - i + 2) : Math.max(0, Math.min(9, i - j + 2)));
      sec.appendChild(buildSpreadsheetGrid({
        rows: criteria, cols: criteria,
        rowLabels: labels, colLabels: labels,
        key:'matrix', defaultFn: defPPF, pair: true,
      }));
      c.appendChild(sec);
    },

    fuzzy_pairwise: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Fuzzy pairwise matrix',
        'each cell is a triangular fuzzy number (l, m, u)'));
      const labels = Array.from({length: criteria}, (_, i) => 'C'+(i+1));
      const defF = (i, j, k) => {
        if (i === j) return 1;
        if (i < j) return [1, 2, 3][k];
        return [(1/3).toFixed(3), 0.5, 1][k];
      };
      sec.appendChild(buildSpreadsheetGrid({
        rows: criteria, cols: criteria,
        rowLabels: labels, colLabels: labels,
        key:'matrix', defaultFn: defF, fuzzy: true,
      }));
      c.appendChild(sec);
    },

    supermatrix: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Super-matrix', 'column-stochastic (cluster-influence)'));
      const labels = Array.from({length: criteria}, (_, i) => 'C'+(i+1));
      const defS = (i, j) => i === j ? 0.5 : (1 / criteria).toFixed(3);
      sec.appendChild(buildSpreadsheetGrid({
        rows: criteria, cols: criteria, rowLabels:labels, colLabels:labels,
        key:'matrix', defaultFn: defS,
      }));
      c.appendChild(sec);
      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    influence_matrix: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Direct-influence matrix',
        '0 = none · 1 = low · 2 = medium · 3 = high · 4 = very high'));
      const labels = Array.from({length: criteria}, (_, i) => 'C'+(i+1));
      const defInf = (i, j) => i === j ? 0 : Math.floor(Math.random() * 5);
      sec.appendChild(buildSpreadsheetGrid({
        rows: criteria, cols: criteria, rowLabels:labels, colLabels:labels,
        key:'matrix', defaultFn: defInf,
      }));
      c.appendChild(sec);
    },

    fuzzy_influence_matrix: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Fuzzy direct-influence matrix',
        'each cell (l, m, u) — diagonal must be (0, 0, 0)'));
      const labels = Array.from({length: criteria}, (_, i) => 'C'+(i+1));
      const defF = (i, j, k) => {
        if (i === j) return 0;
        return [[0,1,2],[1,2,3],[2,3,4]][Math.floor(Math.random()*3)][k];
      };
      sec.appendChild(buildSpreadsheetGrid({
        rows: criteria, cols: criteria, rowLabels:labels, colLabels:labels,
        key:'matrix', defaultFn: defF, fuzzy: true,
      }));
      c.appendChild(sec);
    },

    bwm: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      // best/worst index pickers
      const sec0 = el('div', {class:'input-section'});
      sec0.appendChild(buildSectionHead('Best & worst', 'pick the most and least important criteria'));
      const opts = Array.from({length: criteria}, (_, i) => ({value:i, label:'C'+(i+1)}));
      sec0.appendChild(buildKVRow([
        {key:'best_idx', label:'Best criterion', type:'select', options:opts, value:0},
        {key:'worst_idx', label:'Worst criterion', type:'select', options:opts, value:criteria-1},
      ]));
      c.appendChild(sec0);
      // mic vector — best vs others
      c.appendChild(buildVector({len:criteria, label:'Best-to-others (mic)',
        hint:'Saaty 1–9 — how much more preferred is the best over each criterion',
        valueFn:(j) => j===0 ? 1 : j+1, idPrefix:'mic'}));
      // lic vector — others vs worst
      c.appendChild(buildVector({len:criteria, label:'Others-to-worst (lic)',
        hint:'Saaty 1–9 — how much each criterion dominates the worst',
        valueFn:(j) => criteria-j, idPrefix:'lic'}));
      // params
      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    fuzzy_bwm: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      const sec0 = el('div', {class:'input-section'});
      sec0.appendChild(buildSectionHead('Fuzzy BWM',
        'use linguistic triples — Equally Important (1,1,1), Weakly Important (2/3,1,3/2), ' +
        'Fairly Important (3/2,2,5/2), Very Important (5/2,3,7/2), Absolutely Important (7/2,4,9/2)'));
      const opts = Array.from({length:criteria}, (_, i) => [''+i, 'C'+(i+1)]);
      sec0.appendChild(buildKVRow([
        {key:'best_idx', label:'Best criterion', type:'select', options:opts, value:0},
        {key:'worst_idx', label:'Worst criterion', type:'select', options:opts, value:criteria-1},
      ]));
      c.appendChild(sec0);
      c.appendChild(buildVector({len:criteria, label:'Best-to-others (fuzzy)',
        valueFn:(j, k) => j===0 ? [1,1,1][k] : [[1,2,3,3.5,3.5][Math.min(j,4)],
                                                  [1,3,2,3,4][Math.min(j,4)],
                                                  [1,4,2.5,3.5,4.5][Math.min(j,4)]][k],
        idPrefix:'mic', fuzzy:true}));
      c.appendChild(buildVector({len:criteria, label:'Others-to-worst (fuzzy)',
        valueFn:(j, k) => j===criteria-1 ? [1,1,1][k] :
                         [[3.5,2.5,2,1.5][Math.min(criteria-1-j,3)],
                          [4,3,2,1][Math.min(criteria-1-j,3)],
                          [4.5,3.5,2.5,1.5][Math.min(criteria-1-j,3)]][k],
        idPrefix:'lic', fuzzy:true}));
      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    criteria_rank: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Criteria ranks',
        '1 = most important, 2 = next, ... up to ' + criteria));
      sec.appendChild(buildVector({len:criteria, label:'Ranks',
        valueFn:(j) => j+1, idPrefix:'criteria_rank'}));
      c.appendChild(sec);
    },

    fucom: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      c.appendChild(buildVector({len:criteria, label:'Criterion ranks',
        hint:'1 = most important', valueFn:(j) => j+1, idPrefix:'criteria_rank'}));
      c.appendChild(buildVector({len:criteria, label:'Priority values',
        hint:'comparative priority of consecutive ranks',
        valueFn:(j) => 1 + j*0.5, idPrefix:'criteria_priority'}));
    },

    fuzzy_fucom: (m, c) => {
      const {criteria} = getDims();
      renderDimControls({alts:false, defaults:{criteria}});
      c.appendChild(buildVector({len:criteria, label:'Criterion ranks',
        valueFn:(j) => j+1, idPrefix:'criteria_rank'}));
      c.appendChild(buildVector({len:criteria, label:'Fuzzy priority (l, m, u)',
        valueFn:(j, k) => (1 + j*0.5 + (k-1)*0.1).toFixed(2),
        idPrefix:'criteria_priority', fuzzy:true}));
      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    rafsi: (m, c) => {
      renderDM(m, c);
      const {criteria} = getDims();
      c.appendChild(buildVector({len:criteria, label:'Ideal (optional)',
        valueFn:() => '', idPrefix:'ideal'}));
      c.appendChild(buildVector({len:criteria, label:'Anti-ideal (optional)',
        valueFn:() => '', idPrefix:'anti_ideal'}));
      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    electre_v: (m, c) => {
      renderDM(m, c);
      const {criteria} = getDims();
      c.appendChild(buildVector({len:criteria, label:'Veto thresholds (V)',
        valueFn:(j) => 10, idPrefix:'V'}));
    },

    electre_qpv: (m, c) => {
      renderDM(m, c);
      const {criteria} = getDims();
      c.appendChild(buildVector({len:criteria, label:'Indifference (Q)',
        valueFn:(j) => 1, idPrefix:'Q'}));
      c.appendChild(buildVector({len:criteria, label:'Preference (P)',
        valueFn:(j) => 2, idPrefix:'P'}));
      c.appendChild(buildVector({len:criteria, label:'Veto (V)',
        valueFn:(j) => 10, idPrefix:'V'}));
    },

    electre_qpv_no_w: (m, c) => {
      renderDM(m, c, {noWeights: true});
      const {criteria} = getDims();
      c.appendChild(buildVector({len:criteria, label:'Indifference (Q)', valueFn:()=> demoScalar(1), idPrefix:'Q'}));
      c.appendChild(buildVector({len:criteria, label:'Preference (P)', valueFn:()=> demoScalar(2), idPrefix:'P'}));
      c.appendChild(buildVector({len:criteria, label:'Veto (V)', valueFn:()=> demoScalar(10), idPrefix:'V'}));
    },

    electre_tri: (m, c) => {
      const {alternatives, criteria, classes} = getDims();
      renderDimControls({alts:true, crits:true, classes:true,
                         defaults:{alternatives, criteria, classes}});

      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Decision matrix', `${alternatives} × ${criteria}`));
      matSec.appendChild(buildSpreadsheetGrid({rows:alternatives, cols:criteria,
        key:'matrix', defaultFn: defaultMatrixCell}));
      c.appendChild(matSec);

      c.appendChild(buildVector({len:criteria, label:'Weights',
        valueFn:(j) => demoWeightValue(j, criteria), idPrefix:'weights'}));
      c.appendChild(buildVector({len:criteria, label:'Indifference (Q)', valueFn:()=> demoScalar(1), idPrefix:'Q'}));
      c.appendChild(buildVector({len:criteria, label:'Preference (P)', valueFn:()=> demoScalar(2), idPrefix:'P'}));
      c.appendChild(buildVector({len:criteria, label:'Veto (V)', valueFn:()=> demoScalar(10), idPrefix:'V'}));

      // Boundary profiles: classes-1 rows × criteria cols
      const bSec = el('div', {class:'input-section'});
      bSec.appendChild(buildSectionHead('Boundary profiles (B)',
        `${classes-1} profiles defining the boundaries between ${classes} classes`));
      const labels = Array.from({length:classes-1}, (_, i) => 'B'+(i+1));
      bSec.appendChild(buildSpreadsheetGrid({
        rows: classes-1, cols: criteria, rowLabels: labels,
        key:'B', defaultFn:(i, j) => {
          const col = DEMO.matrix.map(r => r[j] || 0);
          const lo = Math.min(...col), hi = Math.max(...col);
          return (lo + (hi-lo) * (i+1)/classes).toFixed(2);
        }
      }));
      c.appendChild(bSec);
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));

      const ps = el('div', {class:'input-section'});
      ps.appendChild(buildSectionHead('Parameters'));
      ps.appendChild(buildKVRow([{key:'cut_level', label:'Cut level (λ)', type:'number', value:0.75}]));
      c.appendChild(ps);
    },

    electre_tri_central: (m, c) => {
      const {alternatives, criteria, classes} = getDims();
      renderDimControls({alts:true, crits:true, classes:true,
                         defaults:{alternatives, criteria, classes}});
      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Decision matrix', `${alternatives} × ${criteria}`));
      matSec.appendChild(buildSpreadsheetGrid({rows:alternatives, cols:criteria,
        key:'matrix', defaultFn: defaultMatrixCell}));
      c.appendChild(matSec);
      c.appendChild(buildVector({len:criteria, label:'Weights',
        valueFn:(j) => demoWeightValue(j, criteria), idPrefix:'weights'}));
      c.appendChild(buildVector({len:criteria, label:'Indifference (Q)', valueFn:()=> demoScalar(1), idPrefix:'Q'}));
      c.appendChild(buildVector({len:criteria, label:'Preference (P)', valueFn:()=> demoScalar(2), idPrefix:'P'}));

      const cSec = el('div', {class:'input-section'});
      cSec.appendChild(buildSectionHead('Central reference profiles (C)',
        `${classes} central profiles, one per class`));
      const labels = Array.from({length:classes}, (_, i) => 'Ref'+(i+1));
      cSec.appendChild(buildSpreadsheetGrid({
        rows: classes, cols: criteria, rowLabels:labels, key:'C',
        defaultFn:(i, j) => demoProfileValue(i, j, classes, true)
      }));
      c.appendChild(cSec);
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));
      const ps = el('div', {class:'input-section'});
      ps.appendChild(buildSectionHead('Parameters'));
      ps.appendChild(buildKVRow([{key:'cut_level', label:'Cut level', type:'number', value:0.75}]));
      c.appendChild(ps);
    },

    promethee: (m, c) => {
      renderDM(m, c);
      const {criteria} = getDims();
      c.appendChild(buildVector({len:criteria, label:'Indifference (Q)', valueFn:()=> demoScalar(0.5), idPrefix:'Q'}));
      c.appendChild(buildVector({len:criteria, label:'Pre-strict (S)', valueFn:()=> demoScalar(1), idPrefix:'S'}));
      c.appendChild(buildVector({len:criteria, label:'Strict preference (P)', valueFn:()=> demoScalar(2), idPrefix:'P'}));
      // F type per criterion
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Preference functions',
        't1 = usual · t2 = U-shape · t3 = V-shape · t4 = level · t5 = V with indifference · t6 = Gaussian · t7 = U with indifference'));
      const items = [];
      for (let i = 0; i < criteria; i++) {
        items.push({key:`F_${i}`, label:`C${i+1}`, type:'select',
          options:['t1','t2','t3','t4','t5','t6','t7'], value:'t1'});
      }
      sec.appendChild(buildKVRow(items));
      c.appendChild(sec);
      // method-specific params (lmbd, steps)
      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },


    promethee_vi: (m, c) => {
      const {alternatives, criteria} = getDims();
      renderDimControls({defaults:{alternatives, criteria}});
      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Decision matrix', `${alternatives} × ${criteria}`));
      matSec.appendChild(buildSpreadsheetGrid({rows:alternatives, cols:criteria,
        key:'matrix', defaultFn:defaultMatrixCell}));
      c.appendChild(matSec);

      c.appendChild(buildVector({len:criteria, label:'Lower weights',
        hint:'lower bound for each criterion weight',
        valueFn:(j) => isDemoMode() ? Math.max(0.01, Number(demoWeightValue(j, criteria)) * 0.8).toFixed(3) : '',
        idPrefix:'W_lower'}));
      c.appendChild(buildVector({len:criteria, label:'Upper weights',
        hint:'upper bound for each criterion weight',
        valueFn:(j) => isDemoMode() ? Math.min(1, Number(demoWeightValue(j, criteria)) * 1.2).toFixed(3) : '',
        idPrefix:'W_upper'}));
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));
      c.appendChild(buildVector({len:criteria, label:'Indifference (Q)', valueFn:()=> demoScalar(0.5), idPrefix:'Q'}));
      c.appendChild(buildVector({len:criteria, label:'Pre-strict (S)', valueFn:()=> demoScalar(1), idPrefix:'S'}));
      c.appendChild(buildVector({len:criteria, label:'Strict preference (P)', valueFn:()=> demoScalar(2), idPrefix:'P'}));

      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Preference functions',
        't1 = usual · t2 = U-shape · t3 = V-shape · t4 = level · t5 = V with indifference · t6 = Gaussian · t7 = U with indifference'));
      const items = [];
      for (let i = 0; i < criteria; i++) {
        items.push({key:`F_${i}`, label:`C${i+1}`, type:'select',
          options:['t1','t2','t3','t4','t5','t6','t7'], value:'t1'});
      }
      sec.appendChild(buildKVRow(items));
      c.appendChild(sec);

      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    utadis: (m, c) => {
      const {alternatives, criteria, classes} = getDims();
      renderDimControls({alts:true, crits:true, classes:true,
                         defaults:{alternatives, criteria, classes}});

      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Decision matrix', `${alternatives} × ${criteria}`));
      matSec.appendChild(buildSpreadsheetGrid({rows:alternatives, cols:criteria,
        key:'matrix', defaultFn:defaultMatrixCell}));
      c.appendChild(matSec);

      const clsSec = el('div', {class:'input-section'});
      clsSec.appendChild(buildSectionHead('Observed classes', 'one class label per alternative'));
      clsSec.appendChild(buildVector({len:alternatives, label:'Class labels',
        hint:`integers from 1 to ${classes}`,
        valueFn:(i) => {
          if (!isDemoMode()) return '';
          return Math.min(classes, 1 + Math.floor((i * classes) / Math.max(alternatives, 1)));
        },
        idPrefix:'class_labels'}));
      c.appendChild(clsSec);

      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));

      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    flowsort: (m, c) => {
      const {alternatives, criteria, classes} = getDims();
      renderDimControls({alts:true, crits:true, classes:true,
                         defaults:{alternatives, criteria, classes}});
      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Decision matrix', `${alternatives} × ${criteria}`));
      matSec.appendChild(buildSpreadsheetGrid({rows:alternatives, cols:criteria,
        key:'matrix', defaultFn:defaultMatrixCell}));
      c.appendChild(matSec);
      c.appendChild(buildVector({len:criteria, label:'Weights',
        valueFn:(j) => demoWeightValue(j, criteria), idPrefix:'weights'}));
      c.appendChild(buildVector({len:criteria, label:'Indifference (Q)', valueFn:()=> demoScalar(0.5), idPrefix:'Q'}));
      c.appendChild(buildVector({len:criteria, label:'Pre-strict (S)', valueFn:()=> demoScalar(1), idPrefix:'S'}));
      c.appendChild(buildVector({len:criteria, label:'Preference (P)', valueFn:()=> demoScalar(2), idPrefix:'P'}));

      // F type
      const sec = el('div', {class:'input-section'});
      sec.appendChild(buildSectionHead('Preference functions'));
      const items = [];
      for (let i = 0; i < criteria; i++) {
        items.push({key:`F_${i}`, label:`C${i+1}`, type:'select',
          options:['t1','t2','t3','t4','t5','t6','t7'], value:'t1'});
      }
      sec.appendChild(buildKVRow(items));
      c.appendChild(sec);

      // Profiles
      const pSec = el('div', {class:'input-section'});
      pSec.appendChild(buildSectionHead('Class profiles',
        `${classes-1} profiles defining the boundaries between ${classes} classes`));
      const labels = Array.from({length:classes-1}, (_, i) => 'P'+(i+1));
      pSec.appendChild(buildSpreadsheetGrid({
        rows: classes-1, cols: criteria, rowLabels: labels, key: 'profiles',
        defaultFn:(i, j) => demoProfileValue(i, j, classes, false)
      }));
      c.appendChild(pSec);
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));
    },

    cpp_tri: (m, c) => {
      const {alternatives, criteria, classes} = getDims();
      renderDimControls({alts:true, crits:true, classes:true,
                         defaults:{alternatives, criteria, classes}});
      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Decision matrix', `${alternatives} × ${criteria}`));
      matSec.appendChild(buildSpreadsheetGrid({rows:alternatives, cols:criteria,
        key:'matrix', defaultFn:defaultMatrixCell}));
      c.appendChild(matSec);
      c.appendChild(buildVector({len:criteria, label:'Weights',
        valueFn:(j) => demoWeightValue(j, criteria), idPrefix:'weights'}));
      const pSec = el('div', {class:'input-section'});
      pSec.appendChild(buildSectionHead('Class reference profiles', `${classes-1} profiles (quick grid editor)`));
      const labels = Array.from({length:classes-1}, (_, i) => 'P'+(i+1));
      pSec.appendChild(buildSpreadsheetGrid({
        rows: classes-1, cols: criteria, rowLabels:labels, key:'profiles',
        defaultFn:(i, j) => demoProfileValue(i, j, classes, false)
      }));
      c.appendChild(pSec);
      c.appendChild(buildTextAreaSection({
        label:'Profiles / optional structured input',
        hint:'Optional. Paste a Python/JSON-like list or dict of profiles here. When provided, it overrides the grid above.',
        key:'profiles_text',
        value: isDemoMode() ? JSON.stringify({
          'Profile 1': [[5,5,5,5,5,5,5]],
          'Profile 2': [[20,20,20,20,20,20,20],[30,30,30,30,30,30,30]],
          'Profile 3': [[40,40,40,40,40,40,40],[65,65,65,65,65,65,65]],
          'Profile 4': [[70,70,70,85,85,85,85],[85,85,85,85,70,70,70],[75,75,75,75,75,75,75]],
          'Profile 5': [[95,95,95,95,95,95,95]]
        }, null, 2) : ''
      }));
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));
      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    fuzzy_decision_matrix: (m, c) => {
      const {alternatives, criteria} = getDims();
      renderDimControls({defaults:{alternatives, criteria}});
      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Fuzzy decision matrix',
        'each cell is a triangular fuzzy number (l, m, u)'));
      matSec.appendChild(buildSpreadsheetGrid({
        rows: alternatives, cols: criteria, key:'matrix', fuzzy:true,
        defaultFn:(i, j, k) => {
          if (!isDemoMode()) return '';
          const v = DEMO.matrix[i % DEMO.matrix.length][j % DEMO.matrix[0].length];
          return [v*0.9, v, v*1.1][k].toFixed(2);
        }
      }));
      c.appendChild(matSec);
      c.appendChild(buildVector({len:criteria, label:'Fuzzy weights (l, m, u)',
        valueFn:(j, k) => {
          if (!isDemoMode()) return '';
          const v = DEMO.weights[j] || (1/criteria);
          return [v*0.9, v, v*1.1][k].toFixed(3);
        },
        idPrefix:'fuzzy_weights', fuzzy:true}));
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));
      if (m.params && m.params.length) {
        const ps = el('div', {class:'input-section'});
        ps.appendChild(buildSectionHead('Parameters'));
        ps.appendChild(buildKVRow(m.params.map(paramAsKV)));
        c.appendChild(ps);
      }
    },

    fuzzy_weighting_only: (m, c) => {
      const {alternatives, criteria} = getDims();
      renderDimControls({defaults:{alternatives, criteria}});
      const matSec = el('div', {class:'input-section'});
      matSec.appendChild(buildSectionHead('Fuzzy decision matrix',
        'fuzzy weights computed from data — no need to provide weights'));
      matSec.appendChild(buildSpreadsheetGrid({
        rows: alternatives, cols: criteria, key:'matrix', fuzzy:true,
        defaultFn:(i, j, k) => {
          if (!isDemoMode()) return '';
          const v = DEMO.matrix[i % DEMO.matrix.length][j % DEMO.matrix[0].length];
          return [v*0.9, v, v*1.1][k].toFixed(2);
        }
      }));
      c.appendChild(matSec);
      c.appendChild(buildCriterionTypeRow(criteria, isDemoMode() ? DEMO.ctype : null));
    },

    opa: (m, c) => {
      const {alternatives, criteria, experts} = getDims();
      renderDimControls({alts:true, crits:true, experts:true,
                         defaults:{alternatives, criteria, experts}});
      const sec0 = el('div', {class:'input-section'});
      sec0.appendChild(buildSectionHead('OPA — Ordinal Priority Approach',
        `${experts} expert${experts === 1 ? '' : 's'} ranking ${alternatives} alternatives across ${criteria} criteria.`));
      c.appendChild(sec0);
      // expert weights vector (rank order of experts; lower = more authoritative)
      c.appendChild(buildVector({len: experts, label:'Expert ranks (1 = top expert)',
        valueFn:(j) => isDemoMode() ? (j + 1) : '',
        idPrefix:'experts_rank_vec'}));
      // per-expert blocks
      for (let e = 0; e < experts; e++) {
        const block = el('div', {class:'input-section'});
        block.appendChild(buildSectionHead(`Expert ${e+1}`,
          'criterion ranks (top), then per-criterion alternative ranks (bottom)'));
        c.appendChild(block);
        // criteria ranks for this expert
        c.appendChild(buildVector({len: criteria,
          label: `Expert ${e+1} — criteria ranks`,
          valueFn:(j) => isDemoMode() ? (j + 1) : '',
          idPrefix:`opa_crit_${e}`}));
        // alts × criteria for this expert
        const altSec = el('div', {class:'input-section'});
        altSec.appendChild(buildSectionHead(`Expert ${e+1} — alternative ranks per criterion`,
          '1 = best alternative for that criterion'));
        const colLabels = Array.from({length:criteria}, (_, j) => 'C'+(j+1));
        const rowLabels = Array.from({length:alternatives}, (_, i) => 'A'+(i+1));
        altSec.appendChild(buildSpreadsheetGrid({
          rows: alternatives, cols: criteria, rowLabels, colLabels,
          key: `opa_alt_${e}`,
          defaultFn:(i, j) => isDemoMode() ? (i + 1) : ''
        }));
        c.appendChild(altSec);
      }
    },
  };

  // ===================== INPUT COLLECTION =====================
  function collectInputs() {
    const m = state.currentMethod;
    const dims = getDims();
    const fields = $('#input-fields');

    // Generic param collection
    const params = {};
    $$('input[data-key], select[data-key], textarea[data-key]', fields).forEach(inp => {
      const k = inp.dataset.key;
      if (!k) return;
      if (k.startsWith('utility_') || k.startsWith('F_')) return;
      const v = inp.value;
      if (inp.type === 'number') params[k] = parseFloat(v) || 0;
      else params[k] = v;
    });

    // helper: read a labeled section by section title text
    const findSec = (title) => {
      const heads = $$('.input-section-title', fields);
      for (const h of heads) {
        if (h.textContent.toLowerCase().includes(title.toLowerCase())) {
          return h.closest('.input-section');
        }
      }
      return null;
    };

    const payload = {method: m.key};

    const shape = m.shape;

    // ---- gather matrix if any ----
    const matrixSec = findSec('Decision matrix') || findSec('Fuzzy decision matrix') ||
                      findSec('Pairwise comparison') || findSec('PPF pairwise') || findSec('Fuzzy pairwise') ||
                      findSec('Super-matrix') || findSec('Direct-influence') ||
                      findSec('Fuzzy direct-influence') ||
                      findSec('Alternatives ranked per criterion');
    if (matrixSec) {
      const tbody = matrixSec.querySelector('tbody');
      if (tbody && (tbody.dataset.key === 'matrix' || tbody.dataset.key === 'experts_rank_alternatives_T')) {
        const data = readGrid(tbody);
        if (tbody.dataset.key === 'matrix') {
          payload.dataset = data;
          payload.extra_inputs = payload.extra_inputs || {};
          payload.extra_inputs.matrix = data;
        } else {
          // OPA legacy single-expert grid: alts × crits, transpose to crits × alts
          const T = [];
          const nRows = data.length, nCols = data[0].length;
          for (let j = 0; j < nCols; j++) {
            const row = [];
            for (let i = 0; i < nRows; i++) row.push(data[i][j]);
            T.push(row);
          }
          payload.extra_inputs = payload.extra_inputs || {};
          payload.extra_inputs.experts_rank_alternatives = [T];
        }
      }
      // tbodies whose key starts with opa_alt_ / opa_crit_ are handled
      // later by the multi-expert OPA assembler — leave them alone here.
    }

    // ---- weights ----
    const wSec = findSec('Weights') && findSec('Weights') !== findSec('Fuzzy weights') ?
                 findSec('Weights') : null;
    const wSecAll = $$('.input-section', fields).filter(s =>
      s.querySelector('tbody')?.dataset.key === 'weights');
    if (wSecAll.length) {
      const tbody = wSecAll[0].querySelector('tbody');
      payload.weights = readGrid(tbody)[0];
    }

    // ---- criterion type ----
    const ctSec = findSec('Criterion type');
    if (ctSec) payload.criterion_type = readCriterionType(ctSec);

    // ---- generic vector reader: any tbody with a data-key not matrix/weights/ctype ----
    payload.extra_inputs = payload.extra_inputs || {};
    $$('tbody', fields).forEach(tbody => {
      const k = tbody.dataset.key;
      if (!k) return;
      if (['matrix','weights','ctype','experts_rank_alternatives_T'].includes(k)) return;
      const data = readGrid(tbody);
      // Single-row vectors come back as [[...]] — flatten to single dim if rows=1
      if (data.length === 1 && parseInt(tbody.dataset.rows) === 1) {
        payload.extra_inputs[k] = data[0];
      } else {
        payload.extra_inputs[k] = data;
      }
    });

    // ---- params ----
    payload.params = {};
    Object.entries(params).forEach(([k, v]) => {
      if (k.startsWith('best_idx') || k.startsWith('worst_idx')) {
        // BWM specifics: convert to extra_inputs
      } else if (k === 'critical_criterion') {
        payload.extra_inputs.critical_criterion = parseInt(v) || 0;
      } else if (k === 'cut_level') {
        payload.params.cut_level = parseFloat(v);
      } else {
        payload.params[k] = v;
      }
    });
    if (params.best_idx !== undefined) {
      payload.extra_inputs.best = parseInt(params.best_idx);
      payload.extra_inputs.worst = parseInt(params.worst_idx);
    }

    // ---- F (preference function) ----
    const F = [];
    for (let i = 0; ; i++) {
      const sel = fields.querySelector(`select[data-key="F_${i}"]`);
      if (!sel) break;
      F.push(sel.value);
    }
    if (F.length) payload.extra_inputs.F = F;

    // ---- utility_functions ----
    const U = [];
    for (let i = 0; ; i++) {
      const sel = fields.querySelector(`select[data-key="utility_${i}"]`);
      if (!sel) break;
      U.push(sel.value);
    }
    if (U.length) payload.extra_inputs.utility_functions = U;

    // OPA — assemble the multi-expert payload
    if (shape === 'opa') {
      const ei = payload.extra_inputs;
      // Pull the expert weight vector
      const ev = ei.experts_rank_vec;
      ei.experts_rank = Array.isArray(ev) ? ev.map(v => Number(v) || 0)
                                          : (Number(ev) ? [Number(ev)] : [1]);
      delete ei.experts_rank_vec;
      // Walk all opa_crit_e / opa_alt_e tbodies (e is the expert index)
      const critByE = {};
      const altByE = {};
      Object.keys(ei).forEach(k => {
        const mC = k.match(/^opa_crit_(\d+)$/);
        const mA = k.match(/^opa_alt_(\d+)$/);
        if (mC) { critByE[parseInt(mC[1])] = ei[k]; delete ei[k]; }
        else if (mA) { altByE[parseInt(mA[1])] = ei[k]; delete ei[k]; }
      });
      const eCount = Math.max(
        ei.experts_rank.length,
        Object.keys(critByE).length,
        Object.keys(altByE).length, 1);
      const erc = [];
      const era = [];
      for (let e = 0; e < eCount; e++) {
        const critRow = critByE[e] || [];
        erc.push(Array.isArray(critRow) ? critRow.map(v => Number(v) || 0) : [Number(critRow) || 0]);
        // The form already collects alts × crits, which is what opa_method wants.
        const altMat = altByE[e] || [];
        if (Array.isArray(altMat) && altMat.length && Array.isArray(altMat[0])) {
          era.push(altMat.map(row => row.map(v => Number(v) || 0)));
        } else {
          era.push([]);
        }
      }
      ei.experts_rank_criteria = erc;
      ei.experts_rank_alternatives = era;
    }

    if (shape === 'promethee_vi') {
      payload.extra_inputs.W_lower = payload.extra_inputs.W_lower || [];
      payload.extra_inputs.W_upper = payload.extra_inputs.W_upper || [];
      delete payload.weights;
    }

    $$('textarea[data-key]', fields).forEach(inp => {
      const k = inp.dataset.key;
      const raw = inp.value || '';
      if (!raw.trim()) return;
      try {
        payload.extra_inputs[k] = parseLooseStructuredText(raw);
      } catch (err) {
        throw new Error(k + ': ' + err.message);
      }
    });

    if (shape === 'cpp_tri' && payload.extra_inputs.profiles_text) {
      payload.extra_inputs.profiles = payload.extra_inputs.profiles_text;
    }

    if (shape === 'utadis') {
      payload.extra_inputs.class_labels = (payload.extra_inputs.class_labels || []).map(v => parseInt(v, 10) || 1);
    }

    return payload;
  }

  function setGridValues(tbody, data) {
    if (!tbody || !data) return;
    const rows = parseInt(tbody.dataset.rows) || 0;
    const cols = parseInt(tbody.dataset.cols) || 0;
    const fuzzy = tbody.dataset.fuzzy === '1';
    const pair = tbody.dataset.pair === '1';

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const cell = Array.isArray(data[i]) ? data[i][j] : undefined;
        if (fuzzy) {
          const vals = Array.isArray(cell) ? cell : [0, 0, 0];
          for (let k = 0; k < 3; k++) {
            const inp = tbody.querySelector(`input[data-r="${i}"][data-c="${j}"][data-k="${k}"]`);
            if (inp) inp.value = vals[k] ?? '';
          }
        } else if (pair) {
          const vals = Array.isArray(cell) ? cell : [0, 0];
          for (let k = 0; k < 2; k++) {
            const inp = tbody.querySelector(`input[data-r="${i}"][data-c="${j}"][data-k="${k}"]`);
            if (inp) inp.value = vals[k] ?? '';
          }
        } else {
          const inp = tbody.querySelector(`input[data-r="${i}"][data-c="${j}"]`);
          if (inp && cell !== undefined) inp.value = cell;
        }
      }
    }
  }

  function applyExample(example) {
    const ex = example.example || example;
    const payload = ex;
    const dims = ex.dims || {};
    if (dims.alternatives !== undefined) $('#num-alternatives').value = dims.alternatives;
    if (dims.criteria !== undefined) $('#num-criteria').value = dims.criteria;
    if (dims.classes !== undefined) $('#num-classes').value = dims.classes;
    if (dims.experts !== undefined) $('#num-experts').value = dims.experts;

    state.prefillMode = 'empty';
    renderInputForm(state.currentMethod);

    const fields = $('#input-fields');
    const extra = payload.extra_inputs || {};

    // dataset / matrix-like sections
    $$('tbody', fields).forEach(tbody => {
      try {
        const k = tbody.dataset.key;
        if (!k) return;
        if (k === 'matrix') {
          const matrix = extra.matrix || payload.dataset;
          if (matrix) setGridValues(tbody, matrix);
        } else if (k === 'weights' && payload.weights) {
          setGridValues(tbody, [payload.weights]);
        } else if (k === 'fuzzy_weights' && payload.weights) {
          setGridValues(tbody, [payload.weights]);
        } else if (k === 'ctype' && payload.criterion_type) {
          payload.criterion_type.forEach((v, j) => {
            const sel = tbody.querySelector(`select[data-r="0"][data-c="${j}"]`);
            if (sel) sel.value = v;
          });
        } else if (k === 'experts_rank_vec' && Array.isArray(extra.experts_rank)) {
          // Multi-expert OPA: experts_rank vector
          setGridValues(tbody, [extra.experts_rank]);
        } else if (/^opa_crit_(\d+)$/.test(k) && Array.isArray(extra.experts_rank_criteria)) {
          // Multi-expert OPA: per-expert criteria ranks
          const ei = parseInt(k.match(/^opa_crit_(\d+)$/)[1]);
          const row = extra.experts_rank_criteria[ei];
          if (Array.isArray(row)) setGridValues(tbody, [row]);
        } else if (/^opa_alt_(\d+)$/.test(k) && Array.isArray(extra.experts_rank_alternatives)) {
          // Multi-expert OPA: per-expert alts × crits matrix.
          // Both example storage and form layout are alts × crits, so just fill.
          const ei = parseInt(k.match(/^opa_alt_(\d+)$/)[1]);
          const expertMat = extra.experts_rank_alternatives[ei];
          if (Array.isArray(expertMat) && expertMat.length) {
            setGridValues(tbody, expertMat);
          }
        } else if (k === 'experts_rank_criteria' && extra.experts_rank_criteria) {
          const first = Array.isArray(extra.experts_rank_criteria[0]) ? extra.experts_rank_criteria[0] : extra.experts_rank_criteria;
          setGridValues(tbody, [first]);
        } else if (k === 'experts_rank_alternatives_T' && extra.experts_rank_alternatives) {
          const firstExpert = Array.isArray(extra.experts_rank_alternatives[0]) ? extra.experts_rank_alternatives[0] : extra.experts_rank_alternatives;
          setGridValues(tbody, firstExpert);
        } else if (extra[k] !== undefined) {
          const value = extra[k];
          const rows = parseInt(tbody.dataset.rows) || 0;
          const cols = parseInt(tbody.dataset.cols) || 0;
          const fuzzy = tbody.dataset.fuzzy === '1';
          if (rows === 1 && Array.isArray(value)) {
            if (fuzzy && value.length === cols && Array.isArray(value[0])) setGridValues(tbody, [value]);
            else if (!Array.isArray(value[0])) setGridValues(tbody, [value]);
            else setGridValues(tbody, value);
          } else if (Array.isArray(value) && Array.isArray(value[0])) setGridValues(tbody, value);
          else setGridValues(tbody, [value]);
        }
      } catch (err) {
        console.warn('applyExample: failed to populate tbody', tbody.dataset.key, err);
      }
    });

    // scalar / select params
    $$('input[data-key], select[data-key], textarea[data-key]', fields).forEach(inp => {
      const k = inp.dataset.key;
      let v;
      if (payload.params && payload.params[k] !== undefined) {
        v = payload.params[k];
      } else if (extra[k] !== undefined && !['matrix','weights','fuzzy_weights','ctype'].includes(k)) {
        v = extra[k];
      } else {
        return;
      }
      // Textareas (and a few inputs) may receive structured values such as
      // CPP-Tri's profiles dict — pretty-print them readably instead of the
      // useless "[object Object]" toString.
      if (v !== null && typeof v === 'object') {
        try {
          inp.value = formatStructured(v);
        } catch (_) {
          try { inp.value = JSON.stringify(v, null, 2); }
          catch (__) { inp.value = String(v); }
        }
      } else {
        inp.value = v;
      }
    });

    // F selectors
    if (extra.F) {
      extra.F.forEach((v, i) => {
        const sel = fields.querySelector(`select[data-key="F_${i}"]`);
        if (sel) sel.value = v;
      });
    }

    // utility selectors
    if (extra.utility_functions) {
      extra.utility_functions.forEach((v, i) => {
        const sel = fields.querySelector(`select[data-key="utility_${i}"]`);
        if (sel) sel.value = v;
      });
    }
  }

  async function loadExampleForCurrentMethod() {
    if (!state.currentMethod) {
      toast('Pick a method first.');
      return;
    }
    const r = await API.example(state.currentMethod.key);
    if (!r || !r.ok || !r.example) {
      toast('Failed! Something went Wrong...', 'error');
      return;
    }
    try {
      applyExample(r.example);
      toast('Done! Example Loaded', 'success');
    } catch (err) {
      console.error('applyExample failed for', state.currentMethod.key, err);
      toast('Example loaded with warnings — see console.', 'error');
    }
  }


  // ===================== COMPUTE =====================
  async function compute() {
    if (!state.currentMethod) {
      toast('Pick a method first.');
      return;
    }
    let payload;
    try {
      payload = collectInputs();
    } catch (err) {
      toast('Could not read inputs: ' + err.message, 'error');
      console.error(err);
      return;
    }
    showLoading(state.currentMethod.name);
    setStatus('busy', 'Computing');
    const t0 = Date.now();
    try {
      const r = await API.run(payload);
      const dt = Date.now() - t0;
      if (!r.ok) {
        toast('Run failed: ' + r.error, 'error');
        showResultsError(r);
      } else {
        state.lastResult = r;
        renderResults(r);
        toast(`Done in ${dt} ms`, 'success');
        activateTab('results');
      }
    } catch (err) {
      toast('Network error: ' + err.message, 'error');
      console.error(err);
    } finally {
      hideLoading();
      setStatus('ready', 'Ready');
    }
  }

  function setStatus(kind, txt) {
    const pill = $('#status-pill');
    pill.classList.toggle('is-busy', kind === 'busy');
    pill.classList.toggle('is-error', kind === 'error');
    $('#status-text').textContent = txt;
  }

  function showLoading(label) {
    $('#loading-title').textContent = 'Computing ' + (label || '');
    $('#loading-sub').textContent = 'Running the algorithm — please wait.';
    $('#loading-overlay').hidden = false;
  }
  function hideLoading() { $('#loading-overlay').hidden = true; }

  function showResultsError(r) {
    $('#empty-results').hidden = true;
    const c = $('#results-content');
    c.hidden = false;
    c.innerHTML = '';
    c.appendChild(el('div', {class:'console-block', style:{color:'var(--rose)'}},
      'Error: ' + (r.error || 'unknown')));
    if (r.traceback) {
      c.appendChild(el('details', {class:'json-block'},
        el('summary', {}, 'Show traceback'),
        document.createTextNode(r.traceback || r.trace)));
    }
    activateTab('results');
  }

  // ===================== RESULT RENDERERS =====================
  function renderResults(r) {
    $('#empty-results').hidden = true;
    const c = $('#results-content');
    c.hidden = false;
    c.innerHTML = '';

    const kind = r.result_kind || 'ranking';
    const ex = r.extras || {};
    const electreKernelOnly = ['electre_i','electre_i_s','electre_i_v'].includes(r.method);
    const promPartial = r.method === 'promethee_i' || ex.promethee_partial;
    const choiceLike  = ex.choice_only || electreKernelOnly;
    const electreIIIView = !!ex.electre_iii_view;
    const electreIVView  = !!ex.electre_iv_view;
    const opaView        = !!ex.opa_view;

    if (electreIIIView) {
      renderElectreIIIView(c, r);
    } else if (electreIVView) {
      renderElectreIVView(c, r);
    } else if (opaView) {
      renderOpaView(c, r);
    } else if (promPartial) {
      renderProMetheePartial(c, r);
    } else if (kind === 'weights') {
      renderWeightsResult(c, r);
    } else if (kind === 'sorting') {
      renderSortingResult(c, r);
    } else if (kind === 'choice' || choiceLike) {
      renderChoiceResult(c, r);
    } else {
      renderRankingResult(c, r);
    }

    // Method-specific extras (skip the ones the dedicated renderers already cover)
    if (r.extras) {
      if (r.extras.consistency_ratio !== undefined) {
        renderCRMeter(c, r.extras.consistency_ratio);
      }
      if (r.extras.prominence && r.extras.relation) {
        renderInfluenceQuadrant(c, r.extras.prominence, r.extras.relation);
      }
      if (Array.isArray(r.extras.kernel)) {
        renderKernelDisplay(c, r.extras.kernel, r.extras.dominated, r.kernel_size);
      }
      // ELECTRE matrices/partial ranks: only for legacy electre methods (NOT III/IV custom views)
      if (r.method && r.method.startsWith('electre_') && !electreIIIView && !electreIVView) {
        renderElectreMatrices(c, r);
        renderPartialRanks(c, r);
      }
      if (r.method === 'ec_promethee') {
        renderRankFrequency(c, r);
      }
      if (r.method === 'fuzzy_waspas') {
        renderAuxiliaryRankings(c, 'Fuzzy WASPAS components', [
          ['F-WSM', r.extras.f_wsm],
          ['F-WPM', r.extras.f_wpm],
          ['F-WASPAS', r.extras.f_waspas],
        ]);
      }
      if (r.method === 'waspas') {
        renderAuxiliaryRankings(c, 'WASPAS components', [
          ['WSM', r.extras.WSM],
          ['WPM', r.extras.WPM],
          ['WASPAS', r.extras.WASPAS],
        ]);
      }
      if (r.method === 'vikor') {
        renderAuxiliaryRankings(c, 'VIKOR rankings', [
          ['S', r.extras.S],
          ['R', r.extras.R],
          ['Q', r.extras.Q],
        ]);
      }
      if (r.method === 'wisp') {
        renderAuxiliaryRankings(c, 'WISP variants', [
          ['Standard', r.extras.standard_scores],
          ['Simplified', r.extras.simplified_scores],
        ]);
      }
      if (r.method === 'promethee_vi') {
        renderPrometheeViView(c, r);
      }
      if (r.extras.promethee_partial_per_alt && Array.isArray(r.extras.preference_matrix)) {
        renderPartialPerAlt(c, r.extras.preference_matrix);
      }
      if (r.method === 'ahp' && Array.isArray(r.extras.ahp_modes)) {
        renderAhpModesComparison(c, r.extras.ahp_modes, r.extras.selected_mode);
      }
      renderMultimooraBreakdown(c, r);
      renderLaraVisuals(c, r);
      if (r.method === 'promethee_gaia' && r.extras?.gaia_plot) {
        renderImageResult(c, 'PROMETHEE Gaia', r.extras.gaia_plot);
      }
    }

    // Console output
    if (r.stdout && r.stdout.trim()) {
      const sec = el('div', {class:'results-section'});
      sec.appendChild(el('div', {class:'results-section-head'}, 'Console output'));
      sec.appendChild(el('div', {class:'console-block'}, r.stdout));
      c.appendChild(sec);
    }

    // Raw payload
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'Raw output'));
    const det = el('details', {class:'json-block'});
    det.appendChild(el('summary', {}, 'Show JSON'));
    const txt = document.createElement('pre');
    txt.style.margin = 0;
    txt.textContent = JSON.stringify(r, null, 2);
    det.appendChild(txt);
    sec.appendChild(det);
    c.appendChild(sec);
  }

  function classLabel(v) {
    if (v == null || v === '') return '—';
    if (typeof v === 'string') {
      return /^C\d+/i.test(v) ? v.toUpperCase() : (/^\d+$/.test(v) ? ('C' + v) : v);
    }
    return 'C' + String(v);
  }

  function renderChoiceResult(c, r) {
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'Choice result'));
    const kernel = r.extras?.kernel || [];
    const dominated = r.extras?.dominated || [];
    sec.appendChild(el('div', {class:'console-block'},
      `Kernel size: ${kernel.length} · Dominated: ${dominated.length}`));
    c.appendChild(sec);
  }

  function renderImageResult(c, title, base64Png) {
    if (!base64Png) return;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, title));
    sec.appendChild(el('img', {src: 'data:image/png;base64,' + base64Png, class:'result-image', alt: title}));
    c.appendChild(sec);
  }

  function renderRankingResult(c, r) {
    const ranking = r.ranking || [];
    if (!ranking.length) {
      c.appendChild(el('div', {class:'console-block'}, 'No ranking data.'));
      return;
    }
    const sorted = [...ranking].sort((a, b) => a.rank - b.rank);
    const hideTop = !!r.extras?.hide_top;
    const hideScores = !!r.extras?.hide_scores;
    const hideFullRanking = !!r.extras?.hide_full_ranking;
    if (!hideTop) {
      const sec1 = el('div', {class:'results-section'});
      sec1.appendChild(el('div', {class:'results-section-head'}, 'Top alternatives'));
      if (sorted.length >= 3) {
        const podium = el('div', {class:'podium'});
        const medals = ['silver','gold','bronze'];
        const order  = [1, 0, 2];
        order.forEach((idx, k) => {
          const it = sorted[idx];
          if (!it) return;
          podium.appendChild(el('div', {class:'podium-step ' + medals[k]},
            el('div', {class:'podium-rank'}, '#' + it.rank),
            el('div', {class:'podium-name'}, it.alternative_label || ('A'+(it.alternative_index+1))),
            el('div', {class:'podium-score'}, hideScores ? 'rank only' : fmt(it.score))
          ));
        });
        sec1.appendChild(podium);
      }
      c.appendChild(sec1);
    }
    if (hideFullRanking) return;

    const sec2 = el('div', {class:'results-section'});
    sec2.appendChild(el('div', {class:'results-section-head'},
      hideScores ? 'Ranking' : 'Full ranking'));

    if (hideScores) {
      // Compact rank-only list (no bars, no score column) — used for PROMETHEE III.
      const list = el('div', {class:'rank-list'});
      sorted.forEach(it => {
        const rankClass = it.rank <= 3 ? ' r' + it.rank : '';
        list.appendChild(el('div', {class:'rank-item rank-item-compact'},
          el('div', {class:'rank-item-rank' + rankClass}, '#' + it.rank),
          el('div', {class:'rank-item-body'},
            el('div', {class:'rank-item-name'},
              it.alternative_label || ('A'+(it.alternative_index+1))))
        ));
      });
      sec2.appendChild(list);
      c.appendChild(sec2);
      return;
    }

    const list = el('div', {class:'rank-list'});
    const scores = sorted.map(s => Number(s.score) || 0);
    const minS = Math.min(...scores), maxS = Math.max(...scores);
    const signed = (minS < 0 && maxS > 0) || r.extras?.signed_scale;
    const denom = signed ? (Math.max(Math.abs(minS), Math.abs(maxS)) || 1) : ((maxS - minS) || 1);
    sorted.forEach(it => {
      const rankClass = it.rank <= 3 ? ' r' + it.rank : '';
      const score = Number(it.score) || 0;
      const pct = signed ? Math.abs(score) / denom * 50 : ((score - minS) / denom) * 100;
      const fillStyle = signed
        ? {width:'0%', left: (score >= 0 ? '50%' : `${50 - pct}%`)}
        : {width:'0%', left:'0%'};
      const item = el('div', {class:'rank-item'},
        el('div', {class:'rank-item-rank' + rankClass}, '#' + it.rank),
        el('div', {class:'rank-item-body'},
          el('div', {class:'rank-item-name'}, it.alternative_label || ('A'+(it.alternative_index+1))),
          el('div', {class:'rank-item-bar' + (signed ? ' signed' : '')},
            el('div', {class:'rank-item-bar-fill', style:fillStyle}))
        ),
        el('div', {class:'rank-item-score'}, fmt(it.score))
      );
      list.appendChild(item);
      requestAnimationFrame(() => {
        item.querySelector('.rank-item-bar-fill').style.width = pct + '%';
      });
    });
    sec2.appendChild(list);
    c.appendChild(sec2);
  }

  function renderWeightsResult(c, r) {
    const weights = r.weights || [];
    if (!weights.length) {
      c.appendChild(el('div', {class:'console-block'}, 'No weights produced.'));
      return;
    }
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'Computed criterion weights'));
    const total = weights.reduce((a, b) => a + Math.abs(b), 0) || 1;
    const palette = ['#f0b73a','#5fd7d2','#9a8cd6','#6dc18a','#d97676','#ffd06b','#8aece8','#b8b3a5'];
    const wrap = el('div', {class:'weight-viz'});

    // SVG pie
    const SIZE = 240, R = 100, CX = SIZE/2, CY = SIZE/2;
    let acc = 0;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'weight-pie');
    svg.setAttribute('viewBox', `0 0 ${SIZE} ${SIZE}`);
    weights.forEach((w, i) => {
      const frac = Math.abs(w) / total;
      const start = acc * Math.PI * 2 - Math.PI/2;
      const end = (acc + frac) * Math.PI * 2 - Math.PI/2;
      acc += frac;
      const x1 = CX + R * Math.cos(start);
      const y1 = CY + R * Math.sin(start);
      const x2 = CX + R * Math.cos(end);
      const y2 = CY + R * Math.sin(end);
      const large = frac > 0.5 ? 1 : 0;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const d = `M ${CX} ${CY} L ${x1} ${y1} A ${R} ${R} 0 ${large} 1 ${x2} ${y2} Z`;
      path.setAttribute('d', d);
      path.setAttribute('fill', palette[i % palette.length]);
      path.setAttribute('opacity', '0.85');
      path.setAttribute('stroke', '#0c0d11');
      path.setAttribute('stroke-width', '2');
      svg.appendChild(path);
    });
    // donut hole
    const hole = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    hole.setAttribute('cx', CX); hole.setAttribute('cy', CY);
    hole.setAttribute('r', 50); hole.setAttribute('fill', '#0c0d11');
    svg.appendChild(hole);
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', CX); text.setAttribute('y', CY + 5);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', '#f6efe1');
    text.setAttribute('font-family', 'IBM Plex Mono, monospace');
    text.setAttribute('font-size', '12');
    text.textContent = weights.length + ' criteria';
    svg.appendChild(text);
    wrap.appendChild(svg);

    // Bars
    const bars = el('div', {class:'weight-bars'});
    const maxAbs = Math.max(...weights.map(w => Math.abs(w))) || 1;
    weights.forEach((w, i) => {
      const pct = (w / weights.reduce((a, b) => a + b, 0) * 100);
      const bar = el('div', {class:'weight-bar'},
        el('div', {class:'weight-bar-name'}, (r.extras?.weight_labels?.[i]) || ('C' + (i+1))),
        el('div', {class:'weight-bar-track'},
          el('div', {class:'weight-bar-fill',
            style:{width:'0%', background: palette[i%palette.length]}})),
        el('div', {class:'weight-bar-pct'}, (pct.toFixed(1)) + '%')
      );
      bars.appendChild(bar);
      requestAnimationFrame(() => {
        bar.querySelector('.weight-bar-fill').style.width =
          (Math.abs(w)/maxAbs * 100).toFixed(1) + '%';
      });
    });
    wrap.appendChild(bars);
    sec.appendChild(wrap);
    c.appendChild(sec);
  }

  function renderSortingExtras(c, r) {
    return;
  }

  function renderSortingResult(c, r) {
    const cls = r.class_per_alternative || [];
    if (!cls.length) {
      c.appendChild(el('div', {class:'console-block'}, 'No classification produced.'));
      return;
    }
    // Per-alternative class assignments (replaces the "Classification by class"
    // grouping that used to live here — all ELECTRE Tri / FlowSort variants
    // now show one row per alternative instead of class buckets).
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'Assignments'));
    const list = el('div', {class:'rank-list'});
    cls.forEach((cl, i) => {
      list.appendChild(el('div', {class:'class-card-item'},
        el('span', {}, 'A' + (i+1)),
        el('span', {class:'class-card-item-idx'},
          (cl == null) ? 'Unassigned' : classLabel(cl))
      ));
    });
    sec.appendChild(list);
    c.appendChild(sec);

    const ex = r.extras || {};
    if (Array.isArray(ex.lowest_classes) && Array.isArray(ex.highest_classes)) {
      const sec2 = el('div', {class:'results-section'});
      sec2.appendChild(el('div', {class:'results-section-head'}, 'Class intervals'));
      const list = el('div', {class:'rank-list'});
      ex.lowest_classes.forEach((low, i) => {
        const high = ex.highest_classes[i];
        list.appendChild(el('div', {class:'class-card-item'},
          el('span', {}, 'A' + (i+1)),
          el('span', {class:'class-card-item-idx'}, `[${classLabel(low)}, ${classLabel(high)}]`)
        ));
      });
      sec2.appendChild(list);
      c.appendChild(sec2);
    }

    const pc = ex.pessimistic;
    const oc = ex.optimistic;
    if (Array.isArray(pc) && Array.isArray(oc) && pc.length === oc.length) {
      const sec3 = el('div', {class:'results-section'});
      sec3.appendChild(el('div', {class:'results-section-head'}, 'Classification rules'));
      const list = el('div', {class:'rank-list'});
      pc.forEach((p, i) => {
        list.appendChild(el('div', {class:'class-card-item'},
          el('span', {}, 'A' + (i+1)),
          el('span', {class:'class-card-item-idx'}, `PC ${classLabel(p)} · OC/PD ${classLabel(oc[i])}`)
        ));
      });
      sec3.appendChild(list);
      c.appendChild(sec3);
    }

    if (ex.classification_by_rule && typeof ex.classification_by_rule === 'object') {
      const sec4 = el('div', {class:'results-section'});
      sec4.appendChild(el('div', {class:'results-section-head'}, 'All rules'));
      const grid2 = el('div', {class:'classification-grid'});
      Object.entries(ex.classification_by_rule).forEach(([rule, vals]) => {
        if (!Array.isArray(vals)) return;
        const card = el('div', {class:'class-card'}, el('div', {class:'class-card-head'}, rule.toUpperCase()));
        const list = el('div', {class:'class-card-list'});
        vals.forEach((v, i) => {
          list.appendChild(el('div', {class:'class-card-item'},
            el('span', {}, 'A' + (i+1)),
            el('span', {class:'class-card-item-idx'}, classLabel(v))
          ));
        });
        card.appendChild(list);
        grid2.appendChild(card);
      });
      sec4.appendChild(grid2);
      c.appendChild(sec4);
    }

    if (ex.classification_by_mode_rule && typeof ex.classification_by_mode_rule === 'object') {
      const sec5 = el('div', {class:'results-section'});
      sec5.appendChild(el('div', {class:'results-section-head'}, 'All FlowSort mode/rule combinations'));
      const grid3 = el('div', {class:'classification-grid'});
      Object.entries(ex.classification_by_mode_rule).forEach(([name, vals]) => {
        if (!Array.isArray(vals)) return;
        const card = el('div', {class:'class-card'}, el('div', {class:'class-card-head'}, name));
        const list = el('div', {class:'class-card-list'});
        vals.forEach((v, i) => {
          list.appendChild(el('div', {class:'class-card-item'},
            el('span', {}, 'A' + (i+1)),
            el('span', {class:'class-card-item-idx'}, classLabel(v))
          ));
        });
        card.appendChild(list);
        grid3.appendChild(card);
      });
      sec5.appendChild(grid3);
      c.appendChild(sec5);
    }
  }

  function renderAuxiliaryRankings(c, title, entries) {
    const usable = (entries || []).filter(([, vals]) => Array.isArray(vals) && vals.length);
    if (!usable.length) return;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, title));
    const grid = el('div', {class:'classification-grid'});
    usable.forEach(([name, vals]) => {
      const card = el('div', {class:'class-card'}, el('div', {class:'class-card-head'}, name));
      const list = el('div', {class:'class-card-list'});
      vals.forEach((score, i) => list.appendChild(el('div', {class:'class-card-item'},
        el('span', {}, 'A' + (i+1)),
        el('span', {class:'class-card-item-idx'}, fmt(score))
      )));
      card.appendChild(list);
      grid.appendChild(card);
    });
    sec.appendChild(grid);
    c.appendChild(sec);
  }

  function renderElectreMatrices(c, r) {
    const extras = r.extras || {};
    const mats = [
      ['Concordance', extras.concordance],
      ['Discordance', extras.discordance],
      ['Credibility', extras.credibility],
      ['Strong outranking', extras.strong_outranking],
      ['Weak outranking', extras.weak_outranking],
      ['Dominance', extras.dominance_matrix],
      ['Preference matrix', extras.preference_matrix],
    ].filter(([, m]) => Array.isArray(m));
    mats.forEach(([title, m]) => {
      const sec = el('div', {class:'results-section'});
      sec.appendChild(el('div', {class:'results-section-head'}, title));
      const wrap = el('div', {class:'matrix-wrap'});
      const table = el('table', {class:'matrix-table'});
      const thead = el('thead');
      const hr = el('tr');
      hr.appendChild(el('th', {}, ''));
      for (let j = 0; j < m.length; j++) hr.appendChild(el('th', {}, 'A' + (j+1)));
      thead.appendChild(hr);
      table.appendChild(thead);
      const tbody = el('tbody');
      m.forEach((row, i) => {
        const tr = el('tr');
        tr.appendChild(el('th', {}, 'A' + (i+1)));
        (row || []).forEach(v => tr.appendChild(el('td', {}, typeof v === 'number' ? fmt(v) : String(v))));
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      sec.appendChild(wrap);
      c.appendChild(sec);
    });
  }

  // ----- Specialised view renderers (ELECTRE III/IV, OPA, PROMETHEE I/VI) -----
  function _altLabel(idx0) { return 'A' + (idx0 + 1); }

  // Distillation rank ('a3' or 'a3; a5' tie-string -> 'A3' or 'A3, A5')
  function _formatDistillationLevel(level) {
    if (Array.isArray(level)) {
      return level.map(t => _formatDistillationLevel(t)).join(', ');
    }
    if (level == null) return '—';
    const txt = String(level);
    return txt.split(/[;,]/).map(s => {
      const m = s.match(/(\d+)/);
      return m ? 'A' + m[1] : s.trim();
    }).join(', ');
  }

  function renderDistillationCards(c, title, rank_D, rank_A, rank_N) {
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, title));
    const grid = el('div', {class:'classification-grid'});
    const entries = [
      ['Descending (rank_D)', rank_D],
      ['Ascending (rank_A)', rank_A],
      ['Median preorder (rank_N)', rank_N],
    ];
    entries.forEach(([name, val]) => {
      const card = el('div', {class:'class-card'},
        el('div', {class:'class-card-head'}, name));
      const list = el('div', {class:'class-card-list'});
      if (Array.isArray(val) && val.length) {
        val.forEach((level, i) => {
          list.appendChild(el('div', {class:'class-card-item'},
            el('span', {}, 'Level ' + (i+1)),
            el('span', {class:'class-card-item-idx'}, _formatDistillationLevel(level))
          ));
        });
      } else {
        list.textContent = '—';
      }
      card.appendChild(list);
      grid.appendChild(card);
    });
    sec.appendChild(grid);
    c.appendChild(sec);
  }

  function renderMatrix(c, title, m, opts) {
    if (!Array.isArray(m) || !m.length) return;
    const colLabels = opts && opts.colLabels;
    const rowLabels = opts && opts.rowLabels;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, title));
    const wrap = el('div', {class:'matrix-wrap'});
    const table = el('table', {class:'matrix-table'});
    const thead = el('thead');
    const hr = el('tr');
    hr.appendChild(el('th', {}, ''));
    const cols = Array.isArray(m[0]) ? m[0].length : 1;
    for (let j = 0; j < cols; j++) {
      hr.appendChild(el('th', {}, colLabels ? colLabels[j] : 'A' + (j+1)));
    }
    thead.appendChild(hr);
    table.appendChild(thead);
    const tbody = el('tbody');
    m.forEach((row, i) => {
      const tr = el('tr');
      tr.appendChild(el('th', {}, rowLabels ? rowLabels[i] : 'A' + (i+1)));
      (Array.isArray(row) ? row : [row]).forEach(v => {
        let cellTxt;
        if (v == null) cellTxt = '—';
        else if (typeof v === 'number') cellTxt = fmt(v);
        else cellTxt = String(v);
        tr.appendChild(el('td', {}, cellTxt));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    sec.appendChild(wrap);
    c.appendChild(sec);
  }

  function renderElectreIIIView(c, r) {
    const ex = r.extras || {};
    if (ex.fallback) {
      c.appendChild(el('div', {class:'console-block'},
        'Distillation fell back — only global concordance and credibility are available.'));
    }
    renderMatrix(c, 'Global concordance', ex.global_concordance);
    renderMatrix(c, 'Credibility', ex.credibility);
    renderDistillationCards(c, 'ELECTRE III rankings',
      ex.rank_D, ex.rank_A, ex.rank_N);
    renderMatrix(c, 'rank_P (preference matrix)', ex.rank_P);
  }

  function renderElectreIVView(c, r) {
    const ex = r.extras || {};
    if (ex.fallback) {
      c.appendChild(el('div', {class:'console-block'},
        'Distillation fell back — only credibility is available.'));
    }
    renderMatrix(c, 'Credibility', ex.credibility);
    renderDistillationCards(c, 'ELECTRE IV rankings',
      ex.rank_D, ex.rank_A, ex.rank_N);
    renderMatrix(c, 'rank_P (preference matrix)', ex.rank_P);
  }

  function _renderWeightPanel(c, title, weights, labelFn) {
    if (!Array.isArray(weights)) return;
    // Flatten 2-D singletons (e.g. expert_weights = [[0.6, 0.4]] or [[1.0]])
    let flat = weights;
    while (Array.isArray(flat) && flat.length === 1 && Array.isArray(flat[0])) {
      flat = flat[0];
    }
    if (!Array.isArray(flat) || !flat.length) return;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, title));
    const palette = ['#f0b73a','#5fd7d2','#9a8cd6','#6dc18a','#d97676','#ffd06b','#8aece8','#b8b3a5'];
    const bars = el('div', {class:'weight-bars'});
    const total = flat.reduce((a, b) => a + Math.abs(Number(b) || 0), 0) || 1;
    const maxAbs = Math.max(...flat.map(w => Math.abs(Number(w) || 0))) || 1;
    flat.forEach((w, i) => {
      const num = Number(w) || 0;
      const pct = (num / flat.reduce((a, b) => a + (Number(b) || 0), 0) * 100) || 0;
      const bar = el('div', {class:'weight-bar'},
        el('div', {class:'weight-bar-name'}, labelFn(i)),
        el('div', {class:'weight-bar-track'},
          el('div', {class:'weight-bar-fill',
            style:{width:'0%', background: palette[i % palette.length]}})),
        el('div', {class:'weight-bar-pct'}, pct.toFixed(2) + '%')
      );
      bars.appendChild(bar);
      requestAnimationFrame(() => {
        bar.querySelector('.weight-bar-fill').style.width =
          (Math.abs(num) / maxAbs * 100).toFixed(1) + '%';
      });
    });
    sec.appendChild(bars);
    c.appendChild(sec);
  }

  function renderOpaView(c, r) {
    const ex = r.extras || {};
    _renderWeightPanel(c, 'Expert weights (w_e)', ex.expert_weights,
      i => 'Expert ' + (i+1));
    _renderWeightPanel(c, 'Criteria weights (w_c)', ex.criteria_weights,
      i => 'C' + (i+1));
    _renderWeightPanel(c, 'Alternative weights (w_a)', ex.alternative_weights,
      i => 'A' + (i+1));
  }

  function renderProMetheePartial(c, r) {
    // Show just the partial preorder matrix (P+, P-, I, R) with row/col = alts.
    const ex = r.extras || {};
    const m = ex.preference_matrix;
    if (!Array.isArray(m) || !m.length) {
      c.appendChild(el('div', {class:'console-block'},
        'No preference matrix produced.'));
      return;
    }
    const n = m.length;
    const labels = Array.from({length: n}, (_, i) => 'A' + (i+1));
    renderMatrix(c, 'Partial preorder (PROMETHEE I)', m,
      {colLabels: labels, rowLabels: labels});
    // Quick legend
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'Legend'));
    sec.appendChild(el('div', {class:'console-block'},
      'P+ = preferred over · P- = preferred against · I = indifferent · R = incomparable'));
    c.appendChild(sec);
  }

  function renderPrometheeViView(c, r) {
    const ex = r.extras || {};
    const minus = ex.minus_scores || ex.lower_scores;
    const mid   = ex.mid_scores   || ex.favorable_scores;
    const plus  = ex.plus_scores  || ex.upper_scores;
    if (!Array.isArray(minus) || !Array.isArray(mid) || !Array.isArray(plus)) return;

    function _rankRows(name, scores) {
      const order = scores.map((s, i) => ({i, s: Number(s) || 0}))
                          .sort((a, b) => b.s - a.s);
      const card = el('div', {class:'class-card'},
        el('div', {class:'class-card-head'}, name));
      const list = el('div', {class:'class-card-list'});
      order.forEach((it, k) => {
        list.appendChild(el('div', {class:'class-card-item'},
          el('span', {}, '#' + (k+1) + ' · A' + (it.i+1)),
          el('span', {class:'class-card-item-idx'}, fmt(it.s))
        ));
      });
      card.appendChild(list);
      return card;
    }

    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'},
      'PROMETHEE VI rankings (p6_minus / p6 / p6_plus)'));
    const grid = el('div', {class:'classification-grid'});
    grid.appendChild(_rankRows('p6_minus', minus));
    grid.appendChild(_rankRows('p6 (mid)', mid));
    grid.appendChild(_rankRows('p6_plus', plus));
    sec.appendChild(grid);
    c.appendChild(sec);
  }

  function renderAhpModesComparison(c, modes, selectedMode) {
    if (!Array.isArray(modes) || !modes.length) return;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'},
      'AHP weights × consistency by derivation mode'));
    const grid = el('div', {class:'classification-grid'});
    modes.forEach(m => {
      const isSelected = m.mode === selectedMode;
      const card = el('div', {class:'class-card' + (isSelected ? ' is-selected' : '')},
        el('div', {class:'class-card-head'},
          (m.label || m.mode) + (isSelected ? '  ✓' : '')));
      const list = el('div', {class:'class-card-list'});
      if (m.error) {
        list.appendChild(el('div', {class:'class-card-item'},
          el('span', {style:{color:'var(--rose, #d97676)'}}, m.error)));
      } else {
        const cr = m.consistency_ratio;
        const consistent = cr != null && cr <= 0.10;
        const crColor = consistent ? 'var(--mint, #6dc18a)' : 'var(--rose, #d97676)';
        list.appendChild(el('div', {class:'class-card-item'},
          el('span', {}, 'CR'),
          el('span', {class:'class-card-item-idx', style:{color: crColor}},
            cr == null ? '—' : Number(cr).toFixed(4))));
        (m.weights || []).forEach((w, i) => {
          list.appendChild(el('div', {class:'class-card-item'},
            el('span', {}, 'C' + (i+1)),
            el('span', {class:'class-card-item-idx'}, fmt(w))));
        });
      }
      card.appendChild(list);
      grid.appendChild(card);
    });
    sec.appendChild(grid);
    c.appendChild(sec);
  }

  function renderPartialPerAlt(c, preferenceMatrix) {
    if (!Array.isArray(preferenceMatrix) || !preferenceMatrix.length) return;
    const n = preferenceMatrix.length;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'},
      'Partial ranking per alternative'));
    const grid = el('div', {class:'classification-grid'});
    for (let i = 0; i < n; i++) {
      const row = preferenceMatrix[i] || [];
      // Collect each peer's relation
      const dominates = [], dominated = [], indiff = [], incomp = [];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const v = String(row[j] || '').trim();
        const peer = 'A' + (j + 1);
        if (v === 'P+') dominates.push(peer);
        else if (v === 'P-') dominated.push(peer);
        else if (v === 'I') indiff.push(peer);
        else if (v === 'R') incomp.push(peer);
      }
      const card = el('div', {class:'class-card'},
        el('div', {class:'class-card-head'}, 'A' + (i + 1)));
      const list = el('div', {class:'class-card-list'});
      const rel = (label, arr) => list.appendChild(el('div', {class:'class-card-item'},
        el('span', {}, label),
        el('span', {class:'class-card-item-idx'},
          arr.length ? arr.join(', ') : '—')));
      rel('dominates (P+)', dominates);
      rel('dominated (P-)', dominated);
      rel('indifferent (I)', indiff);
      rel('incomparable (R)', incomp);
      card.appendChild(list);
      grid.appendChild(card);
    }
    sec.appendChild(grid);
    c.appendChild(sec);
  }

  // Build an HTML rank-frequency heatmap from rank_samples (iterations × alts).
  // Each row is an alternative, each column a rank position; the cell's shade
  // and number show how often that alternative got that rank.
  function renderRankFrequency(c, r) {
    const samples = r.extras && r.extras.rank_samples;
    if (!Array.isArray(samples) || !samples.length) return;
    const nIter = samples.length;
    const nAlt  = Array.isArray(samples[0]) ? samples[0].length : 0;
    if (!nAlt) return;
    // freq[alt][rank-1] = count
    const freq = Array.from({length: nAlt}, () => new Array(nAlt).fill(0));
    for (let it = 0; it < nIter; it++) {
      const row = samples[it];
      for (let i = 0; i < nAlt; i++) {
        const r1 = row[i];
        if (r1 >= 1 && r1 <= nAlt) freq[i][r1 - 1] += 1;
      }
    }
    // Per-alternative summary: most-likely rank + average rank
    const summary = freq.map((row, i) => {
      let modeRank = 1, modeFreq = -1;
      let sumRank = 0, total = 0;
      row.forEach((cnt, k) => {
        if (cnt > modeFreq) { modeFreq = cnt; modeRank = k + 1; }
        sumRank += cnt * (k + 1);
        total += cnt;
      });
      return { alt: i, modeRank, modeFreq, avgRank: total ? sumRank / total : 0 };
    });
    // Order rows by average rank ascending — clearest visual story
    const order = summary.map(s => s.alt).sort((a, b) =>
      summary[a].avgRank - summary[b].avgRank);

    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'},
      `Rank frequency — ${nIter.toLocaleString()} iterations`));

    const wrap = el('div', {class:'rank-freq-wrap'});
    const table = el('table', {class:'rank-freq-table'});
    const thead = el('thead');
    const hr = el('tr');
    hr.appendChild(el('th', {class:'rank-freq-corner'}, 'Alt \\ Rank'));
    for (let k = 0; k < nAlt; k++) {
      hr.appendChild(el('th', {class:'rank-freq-rank-h'}, '#' + (k + 1)));
    }
    hr.appendChild(el('th', {class:'rank-freq-summary-h'}, 'avg'));
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = el('tbody');
    order.forEach(altIdx => {
      const tr = el('tr');
      tr.appendChild(el('th', {class:'rank-freq-alt'}, 'A' + (altIdx + 1)));
      for (let k = 0; k < nAlt; k++) {
        const cnt = freq[altIdx][k];
        const pct = nIter > 0 ? (cnt / nIter) : 0;
        // Amber alpha keyed to frequency. Subtle floor so empty cells aren't pure black.
        const alpha = pct === 0 ? 0 : (0.10 + 0.80 * pct);
        const td = el('td', {class:'rank-freq-cell'},
          cnt > 0 ? (pct >= 0.005 ? (pct * 100).toFixed(0) + '%' : '<1%') : '');
        td.style.background = `rgba(240, 183, 58, ${alpha.toFixed(3)})`;
        if (alpha > 0.45) td.style.color = '#0c0d11';
        td.title = `A${altIdx + 1} ranked #${k + 1} on ${cnt.toLocaleString()} of ${nIter.toLocaleString()} iterations`;
        tr.appendChild(td);
      }
      tr.appendChild(el('td', {class:'rank-freq-avg'},
        summary[altIdx].avgRank.toFixed(2)));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);

    const legend = el('div', {class:'rank-freq-legend'},
      el('span', {}, 'Lighter cells = higher frequency.'),
      el('span', {class:'rank-freq-legend-bar'}));
    wrap.appendChild(legend);

    sec.appendChild(wrap);
    c.appendChild(sec);
  }

  function renderCRMeter(c, cr) {
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'Consistency'));
    const consistent = Number(cr) <= 0.10;
    const status = consistent ? 'Consistent' : 'Inconsistent';
    const meter = el('div', {class:'cr-meter'});
    meter.appendChild(el('div', {class:'cr-meter-label'},
      el('span', {}, 'Consistency Ratio'),
      el('span', {class:'cr-status ' + (consistent ? 'is-consistent' : 'is-inconsistent')}, status)
    ));
    meter.appendChild(el('div', {class:'cr-meter-value ' + (consistent ? 'is-consistent' : 'is-inconsistent')}, fmt(cr, 4)));
    const note = consistent ? '≤ 0.10' : '> 0.10';
    meter.appendChild(el('div', {class:'cr-meter-note'}, note));
    sec.appendChild(meter);
    c.appendChild(sec);
  }

  function renderInfluenceQuadrant(c, prom, rel) {
    if (!Array.isArray(prom) || !Array.isArray(rel)) return;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'},
      'Cause–effect quadrant (D+R prominence vs D−R relation)'));

    const W = 600, H = 460, M = 50;
    const xs = prom.map(Number), ys = rel.map(Number);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xRange = (xMax - xMin) || 1;
    const yRange = Math.max(Math.abs(yMin), Math.abs(yMax)) || 1;
    const x = (v) => M + (v - xMin) / xRange * (W - 2*M);
    const y = (v) => H/2 - v / yRange * (H/2 - M);

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'influence-chart');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    // axis lines
    const axes = [
      {x1:M, y1:H/2, x2:W-M, y2:H/2},  // x axis at y=0 in data space
      {x1:M, y1:M, x2:M, y2:H-M},     // y axis at left
    ];
    axes.forEach(a => {
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      Object.entries(a).forEach(([k,v]) => ln.setAttribute(k, v));
      ln.setAttribute('stroke', 'rgba(246,239,225,0.2)');
      ln.setAttribute('stroke-width', '1');
      svg.appendChild(ln);
    });
    // axis labels
    const lblX = document.createElementNS('http://www.w3.org/2000/svg','text');
    lblX.setAttribute('x', W/2); lblX.setAttribute('y', H-15);
    lblX.setAttribute('text-anchor','middle');
    lblX.setAttribute('fill', '#8a8678');
    lblX.setAttribute('font-family','IBM Plex Mono, monospace');
    lblX.setAttribute('font-size','11');
    lblX.textContent = 'D + R (prominence)';
    svg.appendChild(lblX);
    const lblY = document.createElementNS('http://www.w3.org/2000/svg','text');
    lblY.setAttribute('x', 16); lblY.setAttribute('y', H/2);
    lblY.setAttribute('fill', '#8a8678');
    lblY.setAttribute('font-family','IBM Plex Mono, monospace');
    lblY.setAttribute('font-size','11');
    lblY.setAttribute('transform', `rotate(-90 16 ${H/2})`);
    lblY.setAttribute('text-anchor','middle');
    lblY.textContent = 'D − R (relation)';
    svg.appendChild(lblY);

    // Quadrant labels
    [
      {t:'CAUSE', x: W-M-50, y: M+15, color:'rgba(240,183,58,0.7)'},
      {t:'EFFECT', x: W-M-50, y: H-M-5, color:'rgba(95,215,210,0.7)'},
    ].forEach(q => {
      const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
      txt.setAttribute('x', q.x); txt.setAttribute('y', q.y);
      txt.setAttribute('fill', q.color);
      txt.setAttribute('font-family','Syne, sans-serif');
      txt.setAttribute('font-size','11');
      txt.setAttribute('font-weight','700');
      txt.setAttribute('letter-spacing','0.18em');
      txt.textContent = q.t;
      svg.appendChild(txt);
    });

    // points
    xs.forEach((vx, i) => {
      const cx = x(vx), cy = y(ys[i]);
      const isCause = ys[i] >= 0;
      const c1 = document.createElementNS('http://www.w3.org/2000/svg','circle');
      c1.setAttribute('cx', cx); c1.setAttribute('cy', cy);
      c1.setAttribute('r', 8);
      c1.setAttribute('fill', isCause ? '#f0b73a' : '#5fd7d2');
      c1.setAttribute('opacity','0.85');
      c1.setAttribute('stroke','#0c0d11');
      c1.setAttribute('stroke-width','2');
      svg.appendChild(c1);
      const lbl = document.createElementNS('http://www.w3.org/2000/svg','text');
      lbl.setAttribute('x', cx + 12); lbl.setAttribute('y', cy + 4);
      lbl.setAttribute('fill', '#f6efe1');
      lbl.setAttribute('font-family','IBM Plex Mono, monospace');
      lbl.setAttribute('font-size','11');
      lbl.textContent = 'C' + (i+1);
      svg.appendChild(lbl);
    });

    sec.appendChild(svg);
    c.appendChild(sec);
  }

  function renderKernelDisplay(c, kernel, dominated, kernelSize) {
    if (!Array.isArray(kernel)) return;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'Kernel and dominated alternatives'));
    const total = (state.lastResult.ranking || []).length || Math.max(...kernel, ...(dominated || []).filter(x => Number.isFinite(x)), -1) + 1;
    const kernelRow = el('div', {class:'kernel-row'},
      el('div', {class:'kernel-row-title'}, 'Kernel'));
    const dominatedRow = el('div', {class:'kernel-row'},
      el('div', {class:'kernel-row-title'}, 'Dominated'));
    const kernelList = el('div', {class:'kernel-list'});
    const dominatedList = el('div', {class:'kernel-list'});
    const inK = new Set(kernel);
    const dom = new Set(dominated || []);
    for (let i = 0; i < total; i++) {
      if (inK.has(i)) kernelList.appendChild(el('span', {class:'kernel-pill in-kernel'}, 'A' + (i+1)));
      if (dom.has(i)) dominatedList.appendChild(el('span', {class:'kernel-pill dominated'}, 'A' + (i+1)));
    }
    if (!kernelList.childNodes.length) kernelList.appendChild(el('span', {class:'kernel-pill'}, '—'));
    if (!dominatedList.childNodes.length) dominatedList.appendChild(el('span', {class:'kernel-pill'}, '—'));
    kernelRow.appendChild(kernelList);
    dominatedRow.appendChild(dominatedList);
    sec.appendChild(kernelRow);
    sec.appendChild(dominatedRow);
    c.appendChild(sec);
  }

  function renderPartialRanks(c, r) {
    const ex = r.extras || {};
    const entries = [
      ['Descending distillation', ex.descending],
      ['Ascending distillation', ex.ascending],
      ['Median / final order', ex.median],
      ['Descending ranks', ex.descending_ranks],
      ['Ascending ranks', ex.ascending_ranks],
      ['Mean ranks', ex.mean_ranks],
    ].filter(([,v]) => v !== undefined);
    if (!entries.length) return;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'Partial ranks'));
    const grid = el('div', {class:'classification-grid'});
    entries.forEach(([title, value]) => {
      const card = el('div', {class:'class-card'},
        el('div', {class:'class-card-head'}, title));
      const list = el('div', {class:'class-card-list'});
      if (Array.isArray(value) && Array.isArray(value[0])) {
        value.forEach((group, i) => {
          list.appendChild(el('div', {class:'class-card-item'},
            el('span', {}, 'Level ' + (i+1)),
            el('span', {class:'class-card-item-idx'}, Array.isArray(group) ? group.join(', ') : String(group))
          ));
        });
      } else if (Array.isArray(value)) {
        value.forEach((group, i) => {
          list.appendChild(el('div', {class:'class-card-item'},
            el('span', {}, 'Level ' + (i+1)),
            el('span', {class:'class-card-item-idx'}, String(group))
          ));
        });
      } else {
        list.textContent = String(value);
      }
      card.appendChild(list);
      grid.appendChild(card);
    });
    sec.appendChild(grid);
    c.appendChild(sec);
  }

  function renderMultimooraBreakdown(c, r) {
    if (r.method !== 'multimoora' || !r.extras) return;
    const parts = [
      ['Ratio system', r.extras.Ratio],
      ['Reference point', r.extras.Reference],
      ['Multiplicative form', r.extras.Multiplicative],
    ].filter(([,v]) => Array.isArray(v));
    if (!parts.length) return;
    const sec = el('div', {class:'results-section'});
    sec.appendChild(el('div', {class:'results-section-head'}, 'MULTIMOORA components'));
    const grid = el('div', {class:'classification-grid'});
    parts.forEach(([title, vals]) => {
      const order = vals.map((score, i) => ({alt:i+1, score:Number(score)}))
        .sort((a,b) => (title === 'Reference point' ? a.score - b.score : b.score - a.score));
      const card = el('div', {class:'class-card'}, el('div', {class:'class-card-head'}, title));
      const list = el('div', {class:'class-card-list'});
      order.forEach((it, idx) => {
        list.appendChild(el('div', {class:'class-card-item'},
          el('span', {}, `#${idx+1} · A${it.alt}`),
          el('span', {class:'class-card-item-idx'}, fmt(it.score))
        ));
      });
      card.appendChild(list);
      grid.appendChild(card);
    });
    sec.appendChild(grid);
    c.appendChild(sec);
  }

  function renderLaraVisuals(c, r) {
    if (r.method === 'lara' && r.extras?.overview_plot) renderImageResult(c, 'LaRa overview', r.extras.overview_plot);
    if (r.method === 'lara' && r.extras?.graph_plot) renderImageResult(c, 'LaRa similarity graph', r.extras.graph_plot);
    if (r.method !== 'lara' || !r.extras || !r.extras.info || !Array.isArray(r.scores) || (r.extras?.overview_plot && r.extras?.graph_plot)) return;
    const info = r.extras.info;
    const order = Array.isArray(r.extras.order) ? r.extras.order : [];
    const scores = r.scores.map(Number);

    const sec1 = el('div', {class:'results-section'});
    sec1.appendChild(el('div', {class:'results-section-head'}, 'LaRa overview'));
    const list = el('div', {class:'rank-list'});
    const maxS = Math.max(...scores), minS = Math.min(...scores), range = (maxS - minS) || 1;
    order.forEach((idx0, pos) => {
      const idx = Number(idx0);
      const score = scores[idx];
      const pct = ((score - minS) / range) * 100;
      const item = el('div', {class:'rank-item'},
        el('div', {class:'rank-item-rank'}, '#' + (pos + 1)),
        el('div', {class:'rank-item-body'},
          el('div', {class:'rank-item-name'}, 'A' + (idx + 1)),
          el('div', {class:'rank-item-bar'}, el('div', {class:'rank-item-bar-fill', style:{width:'0%'}}))
        ),
        el('div', {class:'rank-item-score'}, fmt(score))
      );
      list.appendChild(item);
      requestAnimationFrame(() => {
        item.querySelector('.rank-item-bar-fill').style.width = pct + '%';
      });
    });
    sec1.appendChild(list);
    c.appendChild(sec1);

    const S = info.S_graph || info.S_dense;
    if (!Array.isArray(S)) return;
    const n = Math.min(S.length, 24);
    const W = 720, H = 420, cx = W/2, cy = H/2, rad = Math.min(W, H) * 0.34;
    const pts = Array.from({length:n}, (_, i) => {
      const a = -Math.PI/2 + i * 2*Math.PI / n;
      return {x: cx + rad*Math.cos(a), y: cy + rad*Math.sin(a)};
    });
    const sec2 = el('div', {class:'results-section'});
    sec2.appendChild(el('div', {class:'results-section-head'}, 'LaRa similarity graph'));
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'influence-chart');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const wij = Number((S[i] || [])[j] || 0);
        if (wij <= 0) continue;
        const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        ln.setAttribute('x1', pts[i].x); ln.setAttribute('y1', pts[i].y);
        ln.setAttribute('x2', pts[j].x); ln.setAttribute('y2', pts[j].y);
        ln.setAttribute('stroke', 'rgba(95,215,210,' + Math.min(0.85, 0.12 + wij).toFixed(3) + ')');
        ln.setAttribute('stroke-width', String(0.5 + 2.2 * Math.min(1, wij)));
        svg.appendChild(ln);
      }
    }
    for (let i = 0; i < n; i++) {
      const circ = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circ.setAttribute('cx', pts[i].x); circ.setAttribute('cy', pts[i].y); circ.setAttribute('r', 12);
      circ.setAttribute('fill', 'rgba(240,183,58,0.9)'); circ.setAttribute('stroke', '#0c0d11'); circ.setAttribute('stroke-width', '2');
      svg.appendChild(circ);
      const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      txt.setAttribute('x', pts[i].x); txt.setAttribute('y', pts[i].y + 4); txt.setAttribute('text-anchor', 'middle');
      txt.setAttribute('font-size', '11'); txt.setAttribute('fill', '#050608');
      txt.textContent = 'A' + (i + 1);
      svg.appendChild(txt);
    }
    sec2.appendChild(svg);
    c.appendChild(sec2);
  }

  // ===================== EVENTS =====================
  function attachEvents() {
    // search
    $('#method-search').addEventListener('input', (e) => {
      state.searchQuery = e.target.value;
      renderMethodList();
    });
    // mode toggle
    $$('.sidebar-toggle-btn').forEach(b => {
      b.addEventListener('click', () => {
        $$('.sidebar-toggle-btn').forEach(x => x.classList.remove('is-active'));
        b.classList.add('is-active');
        state.listMode = b.dataset.mode;
        renderMethodList();
      });
    });
    // tabs
    $$('.tab-btn').forEach(b => {
      b.addEventListener('click', () => activateTab(b.dataset.tab));
    });
    // dim steppers
    $$('.step-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const dim = btn.dataset.dim;
        const delta = parseInt(btn.dataset.delta);
        const map = {alternatives:'#num-alternatives',
                     criteria:'#num-criteria',
                     classes:'#num-classes',
                     experts:'#num-experts'};
        const inp = $(map[dim]);
        if (!inp) return;
        const v = parseInt(inp.value) + delta;
        const min = parseInt(inp.min) || 1;
        const max = parseInt(inp.max) || 30;
        inp.value = Math.max(min, Math.min(max, v));
        if (state.currentMethod) renderInputForm(state.currentMethod);
      });
    });
    // dim direct edit
    ['#num-alternatives','#num-criteria','#num-classes','#num-experts'].forEach(sel => {
      const node = $(sel);
      if (!node) return;
      node.addEventListener('change', () => {
        if (state.currentMethod) renderInputForm(state.currentMethod);
      });
    });
    // compute
    $('#btn-compute').addEventListener('click', compute);
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); compute(); }
    });
    // reset
    $('#btn-reset').addEventListener('click', () => {
      state.prefillMode = 'empty';
      if (state.currentMethod) renderInputForm(state.currentMethod);
    });
    // shutdown
    $('#btn-shutdown').addEventListener('click', async () => {
      if (!confirm('Shut down the pyDecision server?')) return;
      await API.shutdown();
      toast('Server stopped. You can close this tab.', 'success');
      setStatus('error', 'Offline');
    });
    // paste modal
    $('#btn-paste').addEventListener('click', () => {
      $('#paste-area').value = '';
      $('#paste-modal').hidden = false;
      setTimeout(() => $('#paste-area').focus(), 50);
    });
    $$('#paste-modal [data-close]').forEach(b => {
      b.addEventListener('click', () => $('#paste-modal').hidden = true);
    });
    $('#btn-apply-paste').addEventListener('click', applyPaste);
    $('#input-fields').addEventListener('paste', handleGridPaste);
    // load packaged example
    $('#btn-fill-demo').addEventListener('click', async () => {
      try {
        await loadExampleForCurrentMethod();
      } catch (err) {
        console.error(err);
        toast('Could not load the packaged example.', 'error');
      }
    });
  }

  function parseSpreadsheetText(text) {
    return text.split(/\r?\n/).map(line => {
      let parts;
      if (line.indexOf('\t') >= 0) parts = line.split('\t');
      else if (line.indexOf(',') >= 0) parts = line.split(',');
      else parts = line.split(/\s+/);
      return parts.map(s => s.trim());
    }).filter(r => r.length && r.some(v => v !== ''));
  }

  function fillGridFromPaste(tbody, startRow, startCol, rows, startK=0) {
    if (!tbody || !rows.length) return 0;
    const maxRows = parseInt(tbody.dataset.rows) || 0;
    const maxCols = parseInt(tbody.dataset.cols) || 0;
    const fuzzy = tbody.dataset.fuzzy === '1';
    let written = 0;

    if (fuzzy) {
      const maxFlatCols = maxCols * 3;
      const startFlat = startCol * 3 + startK;
      rows.forEach((rowVals, i) => {
        const r = startRow + i;
        if (r >= maxRows) return;
        rowVals.forEach((val, j) => {
          const flat = startFlat + j;
          if (flat >= maxFlatCols) return;
          const c = Math.floor(flat / 3);
          const k = flat % 3;
          const inp = tbody.querySelector(`input[data-r="${r}"][data-c="${c}"][data-k="${k}"]`);
          if (inp) {
            inp.value = val;
            written += 1;
          }
        });
      });
      return written;
    }

    rows.forEach((rowVals, i) => {
      const r = startRow + i;
      if (r >= maxRows) return;
      rowVals.forEach((val, j) => {
        const c = startCol + j;
        if (c >= maxCols) return;
        const inp = tbody.querySelector(`input[data-r="${r}"][data-c="${c}"]`);
        if (inp) {
          inp.value = val;
          written += 1;
        }
      });
    });
    return written;
  }

  function handleGridPaste(event) {
    const target = event.target;
    if (!target || target.tagName !== 'INPUT') return;
    const tbody = target.closest('tbody');
    if (!tbody) return;
    const text = event.clipboardData?.getData('text/plain') || '';
    if (!text || (!text.includes('\n') && !text.includes('\t') && !text.includes(','))) return;
    const rows = parseSpreadsheetText(text);
    if (!rows.length) return;
    event.preventDefault();
    const startRow = parseInt(target.dataset.r || '0', 10);
    const startCol = parseInt(target.dataset.c || '0', 10);
    const startK = parseInt(target.dataset.k || '0', 10);
    const written = fillGridFromPaste(tbody, startRow, startCol, rows, startK);
    if (written > 0) toast(`Pasted ${rows.length} row(s) into the grid.`, 'success');
  }

  // Apply paste — into the FIRST visible decision matrix grid
  function applyPaste() {
    const text = $('#paste-area').value.trim();
    if (!text) { $('#paste-modal').hidden = true; return; }
    const rows = parseSpreadsheetText(text);

    if (!rows.length) { $('#paste-modal').hidden = true; return; }

    const fields = $('#input-fields');
    const tbody = fields.querySelector('tbody[data-key="matrix"]') ||
                  fields.querySelector('tbody');
    if (!tbody) { toast('No grid to paste into.', 'error'); return; }

    const newR = rows.length;
    const newC = rows[0].length;
    const isFuzzy = tbody.dataset.fuzzy === '1';
    if (isFuzzy && newC % 3 !== 0) {
      toast(`Fuzzy grid expects 3 numbers per cell — got ${newC} columns total.`, 'error');
      return;
    }

    // Resize dim controls and re-render
    $('#num-alternatives').value = newR;
    $('#num-criteria').value = isFuzzy ? newC/3 : newC;
    state.prefillMode = 'empty';
    renderInputForm(state.currentMethod);

    // Now find the new tbody and fill
    const newBody = $('#input-fields').querySelector('tbody[data-key="matrix"]') ||
                    $('#input-fields').querySelector('tbody');
    if (!newBody) { $('#paste-modal').hidden = true; return; }

    const fuzzy2 = newBody.dataset.fuzzy === '1';
    for (let i = 0; i < newR; i++) {
      for (let j = 0; j < (fuzzy2 ? newC/3 : newC); j++) {
        if (fuzzy2) {
          for (let k = 0; k < 3; k++) {
            const inp = newBody.querySelector(`input[data-r="${i}"][data-c="${j}"][data-k="${k}"]`);
            if (inp) inp.value = rows[i][j*3+k] || '0';
          }
        } else {
          const inp = newBody.querySelector(`input[data-r="${i}"][data-c="${j}"]`);
          if (inp) inp.value = rows[i][j] || '';
        }
      }
    }
    $('#paste-modal').hidden = true;
    toast(`Pasted ${newR}×${fuzzy2 ? newC/3 : newC} matrix.`, 'success');
  }

  // ---- launch ----
  document.addEventListener('DOMContentLoaded', boot);

})();
