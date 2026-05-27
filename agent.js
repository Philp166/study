/* HoloFace — low-poly faceless holographic avatar (vanilla JS, no React) */
(function() {
  const NS = 'http://www.w3.org/2000/svg';
  const BASE_VERTS = [
    [100,18],[70,28],[130,28],[48,48],[152,48],
    [36,78],[164,78],[78,72],[122,72],[100,68],
    [30,110],[170,110],[60,102],[140,102],
    [76,116],[124,116],[86,124],[114,124],
    [100,118],[100,138],
    [38,142],[162,142],[64,150],[136,150],
    [92,152],[108,152],[100,160],
    [80,178],[120,178],[100,182],
    [48,178],[152,178],[70,200],[130,200],
    [88,214],[112,214],[100,222],
  ];
  const TRIS = [
    [0,1,2],[1,3,0],[0,2,4],[1,3,7],[2,4,8],[7,9,0],[8,9,0],[9,7,8],
    [3,5,12],[3,7,12],[7,8,9],[4,6,13],[4,8,13],[12,7,14],[13,8,15],
    [12,14,10],[13,15,11],[14,16,18],[15,17,18],[14,15,18],[16,18,17],
    [14,7,16],[15,8,17],[10,12,20],[11,13,21],
    [10,20,22],[11,21,23],[16,22,18],[17,23,18],[16,22,24],[17,23,25],
    [18,24,19],[18,25,19],[24,19,26],[25,19,26],[24,25,26],
    [22,30,27],[23,31,28],[27,26,28],[27,29,28],[22,24,27],[23,25,28],
    [27,29,32],[28,29,33],[30,27,32],[31,28,33],[30,32,34],[31,33,34],
    [32,34,35],[33,34,35],[22,20,30],[23,21,31],
  ];
  const PRESETS = {
    idle:      { amp:1.2, speed:1.0, glitch:0.02, color:'#00f0ff', secondary:'#6b5cff', label:'IDLE' },
    listening: { amp:3.0, speed:2.4, glitch:0.05, color:'#00f0ff', secondary:'#39d8ff', label:'LISTENING' },
    thinking:  { amp:2.0, speed:1.6, glitch:0.18, color:'#6b5cff', secondary:'#00f0ff', label:'THINKING' },
    speaking:  { amp:2.6, speed:2.0, glitch:0.08, color:'#fcee0a', secondary:'#00f0ff', label:'SPEAKING' },
    alert:     { amp:3.6, speed:3.2, glitch:0.30, color:'#ff2e63', secondary:'#fcee0a', label:'ALERT' },
  };

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }

  function createHoloFace(container, opts = {}) {
    let size = opts.size || 360;
    let state = opts.state || 'idle';
    let audioLevel = opts.audioLevel || 0;

    const wrap = document.createElement('div');
    wrap.className = 'holo-wrap';
    wrap.style.cssText = `position:relative;width:${size}px;height:${size*1.2}px;`;

    const base = document.createElement('div');
    base.style.cssText = `position:absolute;left:50%;bottom:8px;transform:translateX(-50%);width:${size*.85}px;height:${size*.18}px;pointer-events:none;filter:blur(2px);`;
    wrap.appendChild(base);

    const svg = el('svg', { viewBox:'0 0 200 240', width:size, height:size*1.2, style:'overflow:visible;position:absolute;inset:0' });
    wrap.appendChild(svg);

    const defs = el('defs', {}, svg);
    const rg = el('radialGradient', { id:'hf-fill', cx:'50%', cy:'40%', r:'60%' }, defs);
    el('stop', { offset:'0%', 'stop-color':'#00f0ff', 'stop-opacity':'0.35' }, rg);
    el('stop', { offset:'60%', 'stop-color':'#6b5cff', 'stop-opacity':'0.10' }, rg);
    el('stop', { offset:'100%', 'stop-color':'#00f0ff', 'stop-opacity':'0' }, rg);
    const lg = el('linearGradient', { id:'hf-stroke', x1:'0', x2:'0', y1:'0', y2:'1' }, defs);
    el('stop', { offset:'0%', 'stop-color':'#00f0ff' }, lg);
    el('stop', { offset:'100%', 'stop-color':'#6b5cff' }, lg);
    const f1 = el('filter', { id:'hf-glow', x:'-30%', y:'-30%', width:'160%', height:'160%' }, defs);
    el('feGaussianBlur', { stdDeviation:'1.6' }, f1);
    const f2 = el('filter', { id:'hf-glow-s', x:'-30%', y:'-30%', width:'160%', height:'160%' }, defs);
    el('feGaussianBlur', { stdDeviation:'4' }, f2);

    const halo = el('ellipse', { cx:100, cy:120, rx:90, ry:105, fill:'url(#hf-fill)', filter:'url(#hf-glow-s)' }, svg);

    const gFill = el('g', {}, svg);
    const gWire = el('g', { filter:'url(#hf-glow)' }, svg);
    const gVerts = el('g', {}, svg);
    const gRgb = el('g', { opacity:'0.35' }, svg);

    const fillPolys = TRIS.map(() => el('polygon', { fill:'#00f0ff', 'fill-opacity':'0.05', stroke:'none' }, gFill));
    const wirePolys = TRIS.map(() => el('polygon', { fill:'none', stroke:'url(#hf-stroke)', 'stroke-opacity':'0.55', 'stroke-width':'0.45', 'stroke-linejoin':'round' }, gWire));
    const vertDots = BASE_VERTS.map((_, i) => el('circle', { r: i%5===0?1.1:0.7, fill:'#00f0ff', 'fill-opacity':'0.75' }, gVerts));
    const rgbPolys = TRIS.slice(0,18).map(() => el('polygon', { fill:'none', stroke:'#ff2e63', 'stroke-opacity':'0.6', 'stroke-width':'0.4' }, gRgb));

    const scanBand = el('rect', { x:0, width:200, height:14, fill:'#00f0ff', opacity:'0.06' }, svg);

    // HUD
    const hudSvg = el('svg', { viewBox:'0 0 200 240', width:size, height:size*1.2, style:'position:absolute;inset:0;pointer-events:none;overflow:visible' });
    wrap.appendChild(hudSvg);

    const hudTicks = [];
    for (let i = 0; i < 36; i++) {
      const a = (i/36)*Math.PI*2;
      const r1=105, r2=i%9===0?114:109;
      const line = el('line', {
        x1:100+Math.cos(a)*r1, y1:120+Math.sin(a)*r1,
        x2:100+Math.cos(a)*r2, y2:120+Math.sin(a)*r2,
        stroke:'#00f0ff', 'stroke-opacity':i%9===0?0.8:0.25, 'stroke-width':'0.5'
      }, hudSvg);
      hudTicks.push(line);
    }

    const arc1G = el('g', {}, hudSvg);
    const arc1 = el('path', { stroke:'#00f0ff', 'stroke-width':'0.8', fill:'none', opacity:'0.6' }, arc1G);
    const arc2G = el('g', {}, hudSvg);
    const arc2 = el('path', { stroke:'#6b5cff', 'stroke-width':'0.5', fill:'none', opacity:'0.5' }, arc2G);
    const stateLabel = el('text', { x:100, y:6, 'text-anchor':'middle', 'fill-opacity':'0.7', 'font-family':'JetBrains Mono, monospace', 'font-size':'6.5', 'letter-spacing':'0.6' }, hudSvg);

    arc1.setAttribute('d', `M ${100+Math.cos(0)*100} ${120+Math.sin(0)*100} A 100 100 0 0 1 ${100+Math.cos(Math.PI/3)*100} ${120+Math.sin(Math.PI/3)*100}`);
    arc2.setAttribute('d', `M ${100+Math.cos(0)*92} ${120+Math.sin(0)*92} A 92 92 0 0 1 ${100+Math.cos(Math.PI/5)*92} ${120+Math.sin(Math.PI/5)*92}`);

    let raf;
    const t0 = performance.now();

    function frame() {
      const tick = (performance.now() - t0) / 1000;
      const p = PRESETS[state] || PRESETS.idle;
      const t = tick * p.speed;
      const amp = p.amp + audioLevel * 4;

      const verts = BASE_VERTS.map(([x, y], i) => {
        const phase = i * 0.42;
        const dx = Math.sin(t*1.7+phase)*amp*(0.5+(i%3)*0.2);
        const dy = Math.cos(t*1.3+phase*1.4)*amp*(0.4+(i%4)*0.15);
        const br = 1 + Math.sin(t*0.7)*0.012;
        return [100+(x-100+dx)*br, 120+(y-120+dy)*br];
      });

      const glitchOn = ((tick*7)|0)%23===0;
      const gx = glitchOn ? (Math.random()-0.5)*14*p.glitch : 0;
      const gy = glitchOn ? (Math.random()-0.5)*6*p.glitch : 0;
      const tx = `translate(${gx},${gy})`;

      gFill.setAttribute('transform', tx);
      gWire.setAttribute('transform', tx);
      gVerts.setAttribute('transform', tx);

      const fillOp = state==='alert' ? 0.10 : 0.05;

      for (let i = 0; i < TRIS.length; i++) {
        const pts = TRIS[i].map(idx => verts[idx].join(',')).join(' ');
        fillPolys[i].setAttribute('points', pts);
        fillPolys[i].setAttribute('fill', p.color);
        fillPolys[i].setAttribute('fill-opacity', fillOp);
        wirePolys[i].setAttribute('points', pts);
      }

      for (let i = 0; i < vertDots.length; i++) {
        vertDots[i].setAttribute('cx', verts[i][0]);
        vertDots[i].setAttribute('cy', verts[i][1]);
        vertDots[i].setAttribute('fill', p.color);
      }

      scanBand.setAttribute('y', (tick*30)%240);
      scanBand.setAttribute('fill', p.color);

      const showRgb = state==='thinking'||state==='alert';
      gRgb.setAttribute('opacity', showRgb?'0.35':'0');
      if (showRgb) {
        gRgb.setAttribute('transform', `translate(${gx+2},${gy})`);
        for (let i = 0; i < rgbPolys.length; i++) {
          const pts = TRIS[i].map(idx => verts[idx].join(',')).join(' ');
          rgbPolys[i].setAttribute('points', pts);
        }
      }

      base.style.background = `radial-gradient(ellipse at center, ${p.color}55, transparent 70%)`;

      // update gradient colors
      rg.children[0].setAttribute('stop-color', p.color);
      rg.children[2].setAttribute('stop-color', p.color);
      lg.children[0].setAttribute('stop-color', p.color);
      lg.children[1].setAttribute('stop-color', p.secondary);

      // HUD
      const ra = (tick*30)%360;
      arc1G.setAttribute('transform', `rotate(${ra} 100 120)`);
      arc2G.setAttribute('transform', `rotate(${-ra*0.6} 100 120)`);
      arc1.setAttribute('stroke', p.color);
      arc2.setAttribute('stroke', p.secondary);
      stateLabel.textContent = p.label;
      stateLabel.setAttribute('fill', p.color);
      hudTicks.forEach(l => { l.setAttribute('stroke', p.color); });

      raf = requestAnimationFrame(frame);
    }

    container.appendChild(wrap);
    raf = requestAnimationFrame(frame);

    return {
      setState(s) { state = s; },
      setAudioLevel(l) { audioLevel = l; },
      setSize(s) {
        size = s;
        wrap.style.width = size + 'px';
        wrap.style.height = (size*1.2) + 'px';
        svg.setAttribute('width', size);
        svg.setAttribute('height', size*1.2);
        hudSvg.setAttribute('width', size);
        hudSvg.setAttribute('height', size*1.2);
        base.style.width = (size*.85) + 'px';
        base.style.height = (size*.18) + 'px';
      },
      destroy() { cancelAnimationFrame(raf); wrap.remove(); }
    };
  }

  window.createHoloFace = createHoloFace;
  window.HOLO_PRESETS = PRESETS;
})();

/* ── Wake Word Detector ── */
(function() {
  const STORAGE_ENABLED = 'cos_wakeword_enabled';
  const STORAGE_PHRASE  = 'cos_wakeword_phrase';
  const DEFAULT_PHRASE  = 'привет штаб';

  function normalize(text) {
    return text.toLowerCase().replace(/[.,!?;:«»"'()]/g, '').replace(/\s+/g, ' ').trim();
  }

  function createWakeWordDetector(opts) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return null;

    let recognition = null;
    let active = false;
    let enabled = localStorage.getItem(STORAGE_ENABLED) === 'true';
    let phrase = normalize(localStorage.getItem(STORAGE_PHRASE) || DEFAULT_PHRASE);
    const onDetected = opts.onDetected;
    const onStateChange = opts.onStateChange || (() => {});

    function start() {
      if (!enabled || active) return;
      recognition = new SR();
      recognition.lang = 'ru-RU';
      recognition.continuous = true;
      recognition.interimResults = false;

      recognition.onresult = (e) => {
        for (let i = e.resultIndex; i < e.results.length; i++) {
          if (e.results[i].isFinal) {
            const text = normalize(e.results[i][0].transcript);
            if (text.includes(phrase)) {
              stop();
              onDetected(text);
              return;
            }
          }
        }
      };

      recognition.onend = () => {
        if (active) {
          setTimeout(() => { if (active) try { recognition.start(); } catch {} }, 200);
        }
      };

      recognition.onerror = (e) => {
        if (e.error === 'not-allowed') {
          active = false;
          onStateChange('denied');
        }
      };

      try {
        recognition.start();
        active = true;
        onStateChange('listening');
      } catch {}
    }

    function stop() {
      active = false;
      if (recognition) { recognition.abort(); recognition = null; }
      onStateChange('idle');
    }

    function toggle() {
      enabled = !enabled;
      localStorage.setItem(STORAGE_ENABLED, String(enabled));
      if (enabled) start(); else stop();
      return enabled;
    }

    function setPhrase(p) {
      phrase = normalize(p);
      localStorage.setItem(STORAGE_PHRASE, p.trim());
    }

    return {
      start, stop, toggle, setPhrase,
      getPhrase() { return localStorage.getItem(STORAGE_PHRASE) || DEFAULT_PHRASE; },
      isEnabled() { return enabled; },
      isActive()  { return active; },
    };
  }

  window.createWakeWordDetector = createWakeWordDetector;
  window.WAKEWORD_DEFAULT_PHRASE = DEFAULT_PHRASE;
  window.getWakeWordSettings = () => ({
    enabled: localStorage.getItem(STORAGE_ENABLED) === 'true',
    phrase: localStorage.getItem(STORAGE_PHRASE) || DEFAULT_PHRASE,
  });
})();
