"""
互動式 HTML 報告（TradingView Lightweight Charts）。
- 下拉切換 標的 / 時框
- 四類訊號分色標記 + 可勾選開關：
    A高(影線插破等高/前高→看空)  A低(影線插破等低/前低→看多)
    B高(假突破收復→看空)         B低(假跌破收復→看多)
- 每個訊號畫出被獵取的流動性水平線（成形→獵取）
- 右側清單可點擊跳轉
"""
import json
from pathlib import Path

_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>5035 流動性獵取掃描器</title>
<script>__LIB__</script>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:#0e1117; color:#d1d4dc; }
  header { padding:8px 16px; border-bottom:1px solid #222; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  select { background:#161b22; color:#d1d4dc; border:1px solid #30363d; border-radius:6px; padding:5px 9px; }
  .stat { font-size:12px; color:#8b949e; }
  .toggles { display:flex; gap:10px; font-size:12px; flex-wrap:wrap; }
  .toggles label { cursor:pointer; user-select:none; display:flex; align-items:center; gap:4px; }
  .chip { display:inline-block; width:10px; height:10px; border-radius:2px; }
  #main { display:flex; height: calc(100vh - 50px); }
  #chart { flex:1; min-width:0; }
  #side { width:320px; border-left:1px solid #222; overflow-y:auto; }
  .sig { padding:7px 12px; border-bottom:1px solid #1b1f27; cursor:pointer; font-size:12px; }
  .sig:hover { background:#161b22; }
  .sig .d { color:#9aa4b2; }
  .tag { font-size:10px; padding:1px 5px; border-radius:3px; color:#0e1117; font-weight:700; margin-right:5px; }
</style>
</head>
<body>
<header>
  <h1>🎯 流動性獵取掃描器</h1>
  <select id="sel"></select>
  <label style="font-size:12px;color:#9aa4b2">最少觸碰
    <select id="minTouch">
      <option value="1">≥1 (含前高低)</option>
      <option value="2">≥2 (含等高低)</option>
      <option value="3">≥3 (強共識)</option>
      <option value="4">≥4 (極強)</option>
    </select>
  </label>
  <div class="toggles" id="toggles"></div>
  <label style="font-size:12px;color:#9aa4b2;cursor:pointer" title="只顯示確認的訊號（加密=taker 介入；美股=放量）"><input type="checkbox" id="onlyTaker"> 僅⚡確認</label>
  <label style="font-size:12px;color:#9aa4b2;cursor:pointer"><input type="checkbox" id="logScale"> 對數座標</label>
  <span class="stat" id="stat"></span>
</header>
<div id="main">
  <div id="chart"></div>
  <div id="side"><div id="list"></div></div>
</div>
<script>
// 任何未攔截的錯誤都印在頁面上，避免靜默空白
window.onerror = function(msg, src, line, col){
  document.body.innerHTML = '<div style="padding:24px;color:#ef5350;font-family:monospace;white-space:pre-wrap;font-size:14px">'+
    '⚠️ 頁面發生錯誤，無法顯示圖表：\n\n'+msg+'\n位置 '+(src||'')+':'+line+':'+col+'</div>';
  return true;
};
if (typeof LightweightCharts === 'undefined') {
  throw new Error('圖表函式庫(LightweightCharts)未載入');
}
const DATA = __DATA__;

// 四類設定：key、中文名、顏色、箭頭、位置
const CAT = {
  Ahigh: {name:'A·影線插破上方(空)', color:'#ef5350', shape:'arrowDown', pos:'aboveBar'},
  Alow:  {name:'A·影線插破下方(多)', color:'#26a69a', shape:'arrowUp',   pos:'belowBar'},
  Bhigh: {name:'B·假突破收復(空)',   color:'#ff9800', shape:'circle',    pos:'aboveBar'},
  Blow:  {name:'B·假跌破收復(多)',   color:'#42a5f5', shape:'circle',    pos:'belowBar'},
};
const enabled = {Ahigh:true, Alow:true, Bhigh:true, Blow:true};
let minTouch = 2;   // 觸碰數過濾門檻（前高/前低不受此限）
let onlyTaker = false;
function catKey(s){ return s.type + (s.side==='high'?'high':'low'); }
function passTouch(s){ return s.touches >= minTouch; }
function passTaker(s){ return !onlyTaker || s.taker_ok===true; }
function tkTxt(s){ return s.taker_z==null ? 'taker —' : `${s.taker_kind==='放量'?'量':'taker'} ${s.taker_z>=0?'+':''}${s.taker_z.toFixed(2)} ${s.taker_kind}${s.taker_ok?' ⚡':''}`; }

const chart = LightweightCharts.createChart(document.getElementById('chart'), {
  layout:{background:{color:'#0e1117'}, textColor:'#d1d4dc'},
  grid:{vertLines:{color:'#1b1f27'}, horzLines:{color:'#1b1f27'}},
  timeScale:{timeVisible:true, secondsVisible:false, borderColor:'#30363d'},
  rightPriceScale:{borderColor:'#30363d'},
  crosshair:{mode:0},
});
const candle = chart.addCandlestickSeries({
  upColor:'#26a69a', downColor:'#ef5350', borderVisible:false,
  wickUpColor:'#26a69a', wickDownColor:'#ef5350',
});
chart.priceScale('right').applyOptions({scaleMargins:{top:0.05, bottom:0.22}});  // 底部讓給成交量
const volSeries = chart.addHistogramSeries({priceFormat:{type:'volume'}, priceScaleId:'vol',
  lastValueVisible:false, priceLineVisible:false});
chart.priceScale('vol').applyOptions({scaleMargins:{top:0.82, bottom:0}, visible:false});
function volData(candles){
  return candles.map(c=>({time:c.time, value:c.volume||0,
    color:c.close>=c.open?'rgba(38,166,154,.45)':'rgba(239,83,80,.45)'}));
}
let lineSeries = [];
let curKey = null;
function clearLines(){ for(const s of lineSeries) chart.removeSeries(s); lineSeries=[]; }

function draw(){
  const d = DATA[curKey];
  clearLines();
  const sigs = d.signals.filter(s => enabled[catKey(s)] && passTouch(s) && passTaker(s));

  // 流動性水平線（被獵取的那條）
  for(const s of sigs){
    const col = (s.side==='high') ? 'rgba(239,83,80,0.6)' : 'rgba(66,165,245,0.6)';
    const ls = chart.addLineSeries({color:col, lineWidth:2, lastValueVisible:false, priceLineVisible:false, lineStyle:0});
    ls.setData([{time:s.level_start_time, value:s.level_price},{time:s.sweep_time, value:s.level_price}]);
    lineSeries.push(ls);
  }
  // 標記：建立觸碰點(灰色小圈) + 獵取點(分色箭頭)
  const markers = [];
  for(const s of sigs){
    const cat = CAT[catKey(s)];
    // 觸碰點
    for(const ti of (s.touch_idxs||[])){
      const cd = d.candles[ti];
      if(cd) markers.push({time:cd.time, position:cat.pos, color:'#8b949e', shape:'circle', text:'測'});
    }
    // 獵取點（⚡=taker/放量確認）
    markers.push({time:s.sweep_time, position:cat.pos, color:cat.color, shape:cat.shape,
                  text:s.type+(s.taker_ok?'⚡':'')});
  }
  // 同一時間只留一個 marker（避免 lightweight-charts 重複時間報錯）
  const seen = new Set();
  const uniq = [];
  for(const m of markers.sort((a,b)=>a.time-b.time)){
    if(seen.has(m.time)) continue; seen.add(m.time); uniq.push(m);
  }
  candle.setMarkers(uniq);

  // 統計
  const cnt = {Ahigh:0,Alow:0,Bhigh:0,Blow:0};
  for(const s of d.signals) cnt[catKey(s)]++;
  document.getElementById('stat').innerHTML =
    `${d.candles.length} 根 · 顯示 <b>${sigs.length}</b>/${d.signals.length}`;

  // 清單
  const list = document.getElementById('list');
  list.innerHTML = '';
  const sorted = [...sigs].sort((a,b)=>b.sweep_time-a.sweep_time);
  for(const s of sorted){
    const cat = CAT[catKey(s)];
    const div = document.createElement('div');
    div.className='sig';
    const dt = new Date(s.sweep_time*1000).toISOString().slice(0,16).replace('T',' ');
    const src = s.source==='equal' ? `等${s.side==='high'?'高':'低'}×${s.touches}` : `前${s.side==='high'?'高':'低'}`;
    div.innerHTML =
      `<div class="d"><span class="tag" style="background:${cat.color}">${s.type}${s.side==='high'?'↑':'↓'}</span>${dt} UTC</div>`+
      `<div>${src} 線 <b>${s.level_price}</b> · 插破至 ${s.extreme} · ${tkTxt(s)}</div>`;
    div.onclick = ()=>{
      const span = Math.max(s.sweep_time - s.level_start_time, 86400);
      chart.timeScale().setVisibleRange({from:s.level_start_time - span*0.2, to:s.sweep_time + span*0.6});
    };
    list.appendChild(div);
  }
}

function render(key){
  curKey = key;
  chart.priceScale('right').applyOptions({autoScale:true});  // 換標的恢復自動縮放（使用者縮放過會被鎖死→新標的空白）
  candle.setData(DATA[key].candles);
  volSeries.setData(volData(DATA[key].candles));
  draw();
  chart.timeScale().fitContent();
}

// 類型開關
const tg = document.getElementById('toggles');
for(const k of Object.keys(CAT)){
  const lab = document.createElement('label');
  const cb = document.createElement('input'); cb.type='checkbox'; cb.checked=true;
  cb.onchange = ()=>{ enabled[k]=cb.checked; draw(); };
  lab.appendChild(cb);
  const chip = document.createElement('span'); chip.className='chip'; chip.style.background=CAT[k].color;
  lab.appendChild(chip);
  lab.appendChild(document.createTextNode(CAT[k].name));
  tg.appendChild(lab);
}

document.getElementById('minTouch').onchange = (e)=>{ minTouch = parseInt(e.target.value,10); draw(); };
document.getElementById('onlyTaker').onchange = (e)=>{ onlyTaker = e.target.checked; draw(); };
document.getElementById('logScale').onchange = (e)=>{
  chart.priceScale('right').applyOptions({ mode: e.target.checked ? 1 : 0 });  // 1=Logarithmic
};

const sel = document.getElementById('sel');
for(const key of Object.keys(DATA)){
  const o=document.createElement('option'); o.value=key; o.textContent=key.replace('|',' · '); sel.appendChild(o);
}
sel.onchange = ()=>render(sel.value);
render(Object.keys(DATA)[0]);
</script>
</body>
</html>
"""


_LIB_PATH = Path(__file__).parent / "vendor" / "lightweight-charts.standalone.production.js"
_CDN_TAG = ('</script><script src="https://unpkg.com/lightweight-charts@4.1.3'
            '/dist/lightweight-charts.standalone.production.js">')


def build_report(datasets: dict, out_path: str):
    payload = json.dumps(datasets, separators=(",", ":"))
    # 圖表函式庫優先內嵌本地檔（離線可開）；沒有就退回 CDN
    if _LIB_PATH.exists():
        lib = _LIB_PATH.read_text(encoding="utf-8").replace("</script>", "<\\/script>")
    else:
        lib = _CDN_TAG
    html = _HTML.replace("__LIB__", lib).replace("__DATA__", payload)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path
