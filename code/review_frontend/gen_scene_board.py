#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harmony4D-style scene board (cluster confirm + clean refs + target binding).
Usage: python gen_scene_board.py <category>

Read ref_clusters.json + flat/ (refs and targets), derive scenes:
  actors = cluster groups (oversized/low-sim split to singletons, anti chain-merge);
  scenes = video connected components via shared actors; all actors in scene (A/B/C...);
  scene targets = all target images from videos in scene.
UI: ✕ delete ref (instant) / ★ pick clean ref /
  ✕ delete target / click actor label then targets to bind (AB/BC/ABC) /
  undo last delete / export full scene map JSON (incl. deletes).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

MAX_GROUP = 12       # groups larger than this treated as chain-merge; split
MIN_SIM = 0.32       # multi-image groups below avg sim also split

if len(sys.argv) != 2:
    sys.exit("Usage: python gen_scene_board.py <category>")
CATEGORY = sys.argv[1]

ROOT = Path(".") / "data" / "cc0_review_full" / CATEGORY
FLAT = ROOT / "flat"
CLUS = ROOT / "ref_clusters.json"
OUT = ROOT / "scene_board.html"
if not CLUS.exists():
    sys.exit(f"not found: {CLUS}")

data = json.loads(CLUS.read_text())
no_face = data.get("no_face", [])

# 1) actor table (anti chain-merge guard)
actors = []
n_split = 0
for g in data["groups"]:
    ms = g["members"]
    if len(ms) > 1 and (len(ms) > MAX_GROUP or g["avg_sim"] < MIN_SIM):
        n_split += 1
        for m in ms:
            # cross-move empty slot: keep a<i> ids stable (star/bind by aid)
            actors.append({"refs": [] if m.get("moved_to") else [m]})
    else:
        actors.append({"refs": [m for m in ms if not m.get("moved_to")]})
for i, a in enumerate(actors):
    a["id"] = f"a{i}"

# 2) scenes = video components (shared actors merge)
videos = sorted({p.name.split("_r")[0].split("_f")[0] for p in FLAT.glob("*.jpg")})
parent = {v: v for v in videos}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
for a in actors:
    vs = sorted({m["video_id"] for m in a["refs"]})
    for v in vs[1:]:
        if vs[0] in parent and v in parent:
            parent[find(vs[0])] = find(v)

# manual merge override: scene_merges.json = [[video ids,...],...] force one scene
MERGE_FILE = ROOT / "scene_merges.json"
if MERGE_FILE.exists():
    for grp in json.loads(MERGE_FILE.read_text()):
        grp = [v for v in grp if v in parent]
        for v in grp[1:]:
            parent[find(grp[0])] = find(v)
    print(f"applied manual scene merges: {MERGE_FILE.name} ({len(json.loads(MERGE_FILE.read_text()))} groups)")

scene_videos = defaultdict(list)
for v in videos:
    scene_videos[find(v)].append(v)

# 3) targets by video
targets_by_vid = defaultdict(list)
for p in sorted(FLAT.glob("*.jpg")):
    if "_r" not in p.name:
        targets_by_vid[p.name.split("_f")[0]].append(p.name)
noface_by_vid = defaultdict(list)
for fn in no_face:
    noface_by_vid[fn.split("_r")[0]].append(fn)

actors_by_vid = defaultdict(set)
for a in actors:
    for m in a["refs"]:
        actors_by_vid[m["video_id"]].add(a["id"])
actor_map = {a["id"]: a for a in actors}

# 4) scene list (multi-video first, then by target count)
scenes = []
for root, vs in scene_videos.items():
    aids = sorted({aid for v in vs for aid in actors_by_vid[v]}, key=lambda x: int(x[1:]))
    tgts = [t for v in sorted(vs) for t in targets_by_vid[v]]
    nofs = [f for v in sorted(vs) for f in noface_by_vid[v]]
    scenes.append({"videos": sorted(vs), "actor_ids": aids, "targets": tgts, "no_face": nofs})
scenes.sort(key=lambda s: (-len(s["videos"]), -len(s["targets"])))

# scene anchor index: resolve "merge into <cat> N" to anchor (stable across rebuilds).
# Note: after cluster/merge changes, regen all boards twice (index refresh, then cross-embed).
(ROOT / "scene_index.json").write_text(json.dumps([s["videos"][0] for s in scenes]))
xidx = {}
for d in sorted(ROOT.parent.iterdir()):
    f = d / "scene_index.json"
    if d.is_dir() and f.exists():
        xidx[d.name] = json.loads(f.read_text())

COLORS = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 5) generate HTML
cards = []
export_scenes = []
for si, sc in enumerate(scenes):
    labels = {}
    actor_rows = []
    for k, aid in enumerate(sc["actor_ids"]):
        lb = LABELS[k % 26] + ("" if k < 26 else str(k // 26))
        color = COLORS[k % len(COLORS)]
        labels[aid] = lb
        cells = "".join(
            f'<div class="cell ref" data-aid="{aid}" data-fn="{m["thumb"]}">'
            f'<img loading="lazy" src="/cc0img/{CATEGORY}/{m["thumb"]}">'
            f'<span class="x" onclick="delRef(event,this)">✕</span>'
            f'<span class="st" onclick="star(event,this)">★</span>'
            f'<div class="cap">{m["video_id"]}</div></div>'
            for m in actor_map[aid]["refs"])
        actor_rows.append(
            f'<div class="line arow" data-aid="{aid}"><span class="chip" data-aid="{aid}" style="background:{color}" '
            f'onclick="brush(this)">{lb}</span><div class="cells">{cells}</div></div>')
    tgt_cells = "".join(
        f'<div class="cell tgt" data-fn="{fn}" onclick="paint(this)">'
        f'<img loading="lazy" src="/cc0img/{CATEGORY}/{fn}">'
        f'<span class="x" onclick="delTgt(event,this)">✕</span>'
        f'<div class="badges"></div><div class="cap">{fn.split("_f")[1][:-4]}</div></div>'
        for fn in sc["targets"])
    # hide faceless refs; auto-delete on export (deleted_refs)
    vids_disp = " ".join(sc["videos"][:4]) + (f" +{len(sc['videos'])} videos" if len(sc["videos"]) > 4 else "")
    anchor = sc["videos"][0]
    cards.append(
        f'<div class="seq" data-s="{si}" data-anchor="{anchor}">'
        f'<div class="sh"><span class="grab" draggable="true">⠿</span> Scene {si+1}/{len(scenes)} · <code>{vids_disp}</code> · {len(sc["actor_ids"])} actors/{len(sc["targets"])} targets'
        f'<span class="mg">Merge into scene <input class="mi" placeholder="12 or hug 12">'
        f'<button class="mb" onclick="mergeTo(this)">Merge</button>'
        f'<button class="tb" onclick="toTest(this)">Move to test set</button>'
        f'<button class="db" onclick="delScene(this)">Delete scene</button></span></div>'
        f'{"".join(actor_rows)}'
        f'<div class="line tgtl"><span class="lb">Targets</span><div class="cells wrapc">{tgt_cells}</div></div></div>')
    export_scenes.append({"scene": si, "anchor": anchor, "videos": sc["videos"],
                          "actors": [{"id": aid, "label": labels[aid],
                                      "refs": [m["thumb"] for m in actor_map[aid]["refs"]]}
                                     for aid in sc["actor_ids"]],
                          "targets": sc["targets"],
                          "no_face": sc["no_face"]})

# server scene snapshot: /testset aggregates tset from board_state
(ROOT / "scenes_export.json").write_text(json.dumps(export_scenes, ensure_ascii=False))

n_multi_scene = sum(1 for s in scenes if len(s["videos"]) > 1)
html = f"""<!DOCTYPE html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Scene board · {CATEGORY}</title><style>
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,"PingFang SC",sans-serif;background:#f4f5f8;color:#1f2430;padding-bottom:70px}}
.top{{position:sticky;top:0;background:#fff;border-bottom:1px solid #e2e5ea;padding:10px 16px;z-index:10;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
.top b{{font-size:15px}} .hint{{color:#6b7280;font-size:12px;max-width:460px}}
button{{background:#4f7cff;color:#fff;border:0;border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer}}
#cur{{font-size:13px;font-weight:700;padding:4px 10px;border-radius:6px;background:#e5e7eb}}
#dcnt{{color:#ef4444;font-weight:700}}
.seq{{background:#fff;margin:10px;border-radius:12px;border:1px solid #e8eaee;overflow:hidden}}
.seq.dropover{{border:2px dashed #4f7cff;background:#f0f5ff}}
.sh{{padding:8px 14px;font-size:13px;color:#374151;background:#fafbfc;border-bottom:1px solid #eef0f3;display:flex;align-items:center;gap:6px;flex-wrap:wrap}} .sh code{{color:#8b5cf6}}
.sh .grab{{color:#9ca3af;font-size:15px;cursor:grab}}
.sh .mg{{margin-left:auto;display:flex;align-items:center;gap:4px;color:#6b7280;font-size:12px}}
.sh .mi{{width:64px;padding:4px 6px;border:1px solid #d1d5db;border-radius:6px;font-size:13px}}
.sh .mb{{padding:4px 10px;font-size:12px}}
.sh .db{{padding:4px 10px;font-size:12px;background:#ef4444}}
.sh .tb{{padding:4px 10px;font-size:12px;background:#8b5cf6}}
.seq.intest{{border:2px solid #8b5cf6;background:#faf9ff}}
.seq.intest .sh{{background:#f5f3ff}}
.line{{display:flex;align-items:flex-start;gap:10px;padding:8px 14px;border-bottom:1px dashed #f0f2f5}} .line:last-child{{border-bottom:0}}
.line.dim{{opacity:.55}}
.lb{{flex:0 0 46px;font-size:12px;color:#6b7280;text-align:right;padding-top:60px}}
.chip{{flex:0 0 46px;height:46px;border-radius:10px;color:#fff;font-size:20px;font-weight:800;display:flex;align-items:center;justify-content:center;cursor:pointer;border:3px solid transparent;margin-top:50px}}
.chip.on{{border-color:#111;box-shadow:0 0 0 3px #fff inset}}
.cells{{display:flex;gap:8px;overflow-x:auto;flex:1}} .cells.wrapc{{flex-wrap:wrap;overflow:visible}}
.cell{{position:relative;flex:0 0 auto;border-radius:8px;overflow:hidden;border:3px solid transparent}}
.cell img{{height:150px;max-width:200px;object-fit:contain;background:#111;display:block;border-radius:5px}}
.cell.tgt img{{height:185px;max-width:260px;cursor:pointer}}
.cell .cap{{font-size:9px;color:#9ca3af;text-align:center;padding:1px}}
.cell .x,.cell .st{{position:absolute;width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;cursor:pointer;font-size:12px}}
.cell .x{{top:4px;right:4px;background:#ef4444;color:#fff}}
.cell .st{{top:4px;left:4px;background:#fff;color:#d1d5db;border:1px solid #e5e7eb}}
.cell.starred .st{{background:#fbbf24;color:#fff;border-color:#f59e0b}}
.cell.starred{{border-color:#fbbf24}}
.badges{{position:absolute;bottom:16px;left:4px;display:flex;gap:3px}}
.badges span{{width:24px;height:24px;border-radius:6px;color:#fff;font-size:13px;font-weight:800;display:flex;align-items:center;justify-content:center}}
.nosel img{{height:110px}}
</style></head><body>
<div class=top><b>Scene board · {CATEGORY}</b><span id=cur>Brush: none</span>
<span class=hint>①✕ delete hides ②★ clean ref ③ brush bind ④ merge: number or "hug 12" ⑤ purple test-set (assigned <span id=tcnt style="color:#8b5cf6;font-weight:700">0</span>) ⑥ export</span>
<a href="/testset" style="font-size:13px">🧪 Test set overview</a>
<button onclick=undo() style="background:#f59e0b">Undo delete</button>
<button onclick=undoMerge() style="background:#8b5cf6">Undo scene merge</button>
<button onclick=exp()>Export scene map</button>
<button id=doneBtn onclick=markDone() style="background:#9ca3af">Mark done</button>
<button onclick="if(confirm('Clear all actions?'))clr()" style="background:#9ca3af">Clear</button></div>
{''.join(cards)}
<script>
const K='cc0scene_{CATEGORY}';
const nOps=s=>((s.del||[]).length+(s.dtgt||[]).length+Object.keys(s.star||{{}}).length+Object.keys(s.bind||{{}}).length+(s.smerge||[]).length+(s.tset||[]).length);
let st=JSON.parse(localStorage.getItem(K)||'{{"del":[],"dtgt":[],"star":{{}},"bind":{{}},"stack":[],"smerge":[],"tset":[]}}');
try{{                       // server shared state: one editor recommended
  const xq=new XMLHttpRequest();xq.open('GET','/cc0state/{CATEGORY}',false);xq.send();
  if(xq.status===200){{const sv=JSON.parse(xq.responseText||'{{}}');if(nOps(sv)>=nOps(st))st=sv;}}
}}catch(e){{}}
let delRefs=new Set(st.del), delTgts=new Set(st.dtgt), starMap=st.star||{{}}, bindMap=st.bind||{{}}, stack=st.stack||[];
let sceneMerges=st.smerge||[];            // in-cat:[src,dst] cross:[src,dstAnchor,dstCat] in order
let testSet=new Set(st.tset||[]);         // test-set scene anchors (whole scene leaves training)
const CAT='{CATEGORY}';
const XIDX={json.dumps(xidx, ensure_ascii=False)};   // category scene number -> anchor (snapshot at gen)
let curBrush=null;
const chipColor={{}};
document.querySelectorAll('.chip').forEach(c=>chipColor[c.dataset.aid]=c.style.background);
let pushT=null;
function pushSrv(){{fetch('/cc0state/{CATEGORY}',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:localStorage.getItem(K)}}).catch(()=>{{}});}}
function save(){{localStorage.setItem(K,JSON.stringify({{del:[...delRefs],dtgt:[...delTgts],star:starMap,bind:bindMap,stack:stack,smerge:sceneMerges,tset:[...testSet]}}));clearTimeout(pushT);pushT=setTimeout(pushSrv,400);}}
const LB='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
function lbl(k){{return LB[k%26]+(k<26?'':Math.floor(k/26));}}

// ===== scene drag merge =====
function applyMergeDOM(src,dst){{
  const s=document.querySelector('.seq[data-anchor="'+src+'"]');
  const d=document.querySelector('.seq[data-anchor="'+dst+'"]');
  if(!s||!d||s===d||s.style.display==='none')return false;
  const dtl=d.querySelector('.line.tgtl');
  let k=d.querySelectorAll('.line.arow').length;
  [...s.querySelectorAll('.line.arow')].forEach(row=>{{
    row.querySelector('.chip').textContent=lbl(k++);
    dtl.before(row);
  }});
  const dcells=dtl.querySelector('.cells');
  [...s.querySelectorAll('.line.tgtl .cell')].forEach(c=>dcells.appendChild(c));
  s.style.display='none';
  return true;
}}
function resolveAnchor(a){{
  let cur=a;const seen=new Set();
  for(;;){{const m=sceneMerges.find(p=>p.length===2&&p[0]===cur);if(!m||seen.has(cur))return cur;seen.add(cur);cur=m[1];}}
}}
function xMerged(a){{return sceneMerges.some(m=>m.length===3&&m[0]===a);}}
function mergeTo(btn){{
  const card=btn.closest('.seq');const inp=card.querySelector('.mi');
  const raw=(inp.value||'').trim();
  if(!raw)return alert('Enter target scene number, or "category number" (e.g. hug 12)');
  let m=raw.match(/^(\\d+)$/), cat=CAT, n;
  if(m){{n=parseInt(m[1]);}}
  else{{
    m=raw.match(/^([a-z_][a-z_0-9]*)[\\s::]+(\\d+)$/i);
    if(!m)return alert('Format: 12 or hug 12');
    cat=m[1].toLowerCase();n=parseInt(m[2]);
  }}
  const src=card.dataset.anchor;
  if(cat===CAT){{
    const t=document.querySelector('.seq[data-s="'+(n-1)+'"]');
    if(!t)return alert('No scene '+n);
    const dst=resolveAnchor(t.dataset.anchor);
    if(dst===src)return alert('Cannot merge into self (or already same scene)');
    if(xMerged(dst))return alert('Target scene already merged into another category');
    if(applyMergeDOM(src,dst)){{sceneMerges.push([src,dst]);save();render();inp.value='';}}
  }}else{{
    if(!XIDX[cat])return alert('No category '+cat+'\\nAvailable: '+Object.keys(XIDX).join(' '));
    if(!XIDX[cat][n-1])return alert(cat+' has only '+XIDX[cat].length+' scenes');
    if(!confirm('Move whole scene into '+cat+' scene '+n+'? Refs/targets/actions move together.'))return;
    btn.disabled=true;btn.textContent='Moving…';
    fetch('/cc0state/{CATEGORY}',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:localStorage.getItem(K)}})
      .then(()=>fetch('/cc0xmove',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{src:CAT,anchor:src,dst:cat,n:n}})}}))
      .then(r=>r.json()).then(j=>{{
        if(!j.ok)throw j.err;
        localStorage.setItem(K,JSON.stringify(j.src_state));   // server stripped moved portion
        alert('Moved into '+cat+' scene '+(j.dst_scene||n)+' — open target board to continue');
        location.href='/cc0scene/'+cat;
      }}).catch(e=>{{alert('Move failed: '+e);btn.disabled=false;btn.textContent='Merge';}});
  }}
}}
let dragSrc=null;
document.querySelectorAll('.seq .grab').forEach(h=>{{
  h.addEventListener('dragstart',e=>{{dragSrc=h.closest('.seq').dataset.anchor;e.dataTransfer.effectAllowed='move';}});
}});
document.querySelectorAll('.seq').forEach(card=>{{
  card.addEventListener('dragover',e=>{{
    if(dragSrc&&card.dataset.anchor!==dragSrc){{e.preventDefault();card.classList.add('dropover');}}
  }});
  card.addEventListener('dragleave',()=>card.classList.remove('dropover'));
  card.addEventListener('drop',e=>{{
    e.preventDefault();card.classList.remove('dropover');
    const dst=card.dataset.anchor;
    if(!dragSrc||dst===dragSrc)return;
    if(xMerged(dst)){{dragSrc=null;return alert('Target scene already merged into another category');}}
    if(applyMergeDOM(dragSrc,dst)){{sceneMerges.push([dragSrc,dst]);save();render();}}
    dragSrc=null;
  }});
}});
function undoMerge(){{
  if(!sceneMerges.length)return alert('No scene merge to undo');
  sceneMerges.pop();save();location.reload();     // reload replays remaining merges
}}
function hiddenActors(){{
  const gone=new Set();
  document.querySelectorAll('.line.arow').forEach(row=>{{
    const refs=[...row.querySelectorAll('.cell.ref')];
    if(refs.every(c=>delRefs.has(c.dataset.fn)))gone.add(row.dataset.aid);
  }});
  return gone;
}}
function render(){{
  const gone=hiddenActors();
  document.querySelectorAll('.cell.ref').forEach(c=>{{
    c.style.display=delRefs.has(c.dataset.fn)?'none':'';
    c.classList.toggle('starred',starMap[c.dataset.aid]===c.dataset.fn);
  }});
  document.querySelectorAll('.line.arow').forEach(row=>{{row.style.display=gone.has(row.dataset.aid)?'none':'';}});
  document.querySelectorAll('.cell.tgt').forEach(c=>{{
    c.style.display=delTgts.has(c.dataset.fn)?'none':'';
    const arr=(bindMap[c.dataset.fn]||[]).filter(aid=>!gone.has(aid));
    c.querySelector('.badges').innerHTML=arr.map(aid=>{{
      const chip=document.querySelector('.chip[data-aid="'+aid+'"]');
      return '<span style="background:'+(chipColor[aid]||'#666')+'">'+(chip?chip.textContent:'?')+'</span>';
    }}).join('');
  }});
  document.querySelectorAll('.seq').forEach(card=>{{     // hide empty cards
    if(xMerged(card.dataset.anchor))return;              // keep legacy merged-away cards hidden
    const any=[...card.querySelectorAll('.cell')].some(c=>c.style.display!=='none');
    card.style.display=any?'':'none';
  }});
  if(curBrush&&gone.has(curBrush))curBrush=null;
  document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on',c.dataset.aid===curBrush));
  const cb=document.querySelector('.chip[data-aid="'+curBrush+'"]');
  document.getElementById('cur').textContent='Brush: '+(cb?cb.textContent:'none');
  document.getElementById('cur').style.background=cb?chipColor[curBrush]:'#e5e7eb';
  document.getElementById('cur').style.color=cb?'#fff':'#1f2430';
  document.getElementById('dcnt').textContent=delRefs.size+delTgts.size;
  document.getElementById('mcnt').textContent=sceneMerges.length;
  document.querySelectorAll('.seq').forEach(card=>{{
    const on=testSet.has(card.dataset.anchor);
    card.classList.toggle('intest',on);
    const tb=card.querySelector('.tb');
    if(tb){{tb.textContent=on?'✓ Test set (click undo)':'Move to test set';tb.style.background=on?'#22a06b':'#8b5cf6';}}
  }});
  document.getElementById('tcnt').textContent=testSet.size;
}}
function toTest(btn){{const a=btn.closest('.seq').dataset.anchor;
  testSet.has(a)?testSet.delete(a):testSet.add(a);save();render();}}
function delRef(e,el){{e.stopPropagation();const c=el.closest('.cell');const fn=c.dataset.fn;
  delRefs.add(fn);stack.push(['ref',fn]);
  if(starMap[c.dataset.aid]===fn)delete starMap[c.dataset.aid];save();render();}}
function delTgt(e,el){{e.stopPropagation();const fn=el.closest('.cell').dataset.fn;
  delTgts.add(fn);stack.push(['tgt',fn]);save();render();}}
function undo(){{const last=stack.pop();if(!last)return alert('Nothing to undo');
  if(last[0]==='scene'){{last[1].forEach(fn=>delRefs.delete(fn));last[2].forEach(fn=>delTgts.delete(fn));}}
  else last[0]==='ref'?delRefs.delete(last[1]):delTgts.delete(last[1]);
  save();render();}}
function delScene(btn){{
  const card=btn.closest('.seq');
  const refs=[...card.querySelectorAll('.cell.ref')].filter(c=>c.style.display!=='none'&&!delRefs.has(c.dataset.fn));
  const tgts=[...card.querySelectorAll('.cell.tgt')].filter(c=>c.style.display!=='none'&&!delTgts.has(c.dataset.fn));
  if(!refs.length&&!tgts.length)return alert('Scene already empty');
  if(!confirm('Delete entire scene?\\n'+refs.length+' refs + '+tgts.length+' targets\\n(Click "Undo delete" to restore whole scene)'))return;
  const br=[],bt=[];
  refs.forEach(c=>{{const fn=c.dataset.fn;delRefs.add(fn);br.push(fn);
    if(starMap[c.dataset.aid]===fn)delete starMap[c.dataset.aid];}});
  tgts.forEach(c=>{{const fn=c.dataset.fn;delTgts.add(fn);bt.push(fn);}});
  stack.push(['scene',br,bt]);save();render();
}}
function star(e,el){{e.stopPropagation();const c=el.closest('.cell');
  starMap[c.dataset.aid]=starMap[c.dataset.aid]===c.dataset.fn?undefined:c.dataset.fn;
  if(!starMap[c.dataset.aid])delete starMap[c.dataset.aid];save();render();}}
function brush(el){{curBrush=curBrush===el.dataset.aid?null:el.dataset.aid;render();}}
function paint(c){{if(!curBrush)return alert('Click an actor label to activate brush');
  const fn=c.dataset.fn;let arr=bindMap[fn]||[];
  arr.includes(curBrush)?arr=arr.filter(x=>x!==curBrush):arr.push(curBrush);
  if(arr.length)bindMap[fn]=arr;else delete bindMap[fn];save();render();}}
function clr(){{delRefs.clear();delTgts.clear();starMap={{}};bindMap={{}};stack=[];sceneMerges=[];testSet.clear();curBrush=null;save();location.reload();}}
// ===== category done flag (reviewed_done; green on /cc0 and home) =====
function setDoneUI(v){{const b=document.getElementById('doneBtn');
  b.textContent=v?'✓ Done':'Mark done';b.style.background=v?'#22a06b':'#9ca3af';b.dataset.on=v?'1':'';}}
fetch('/cc0done/{CATEGORY}').then(r=>r.json()).then(j=>setDoneUI(j.done)).catch(()=>{{}});
function markDone(){{
  const on=document.getElementById('doneBtn').dataset.on==='1';
  if(!confirm(on?'Remove done mark?':'Mark {CATEGORY} done? Shows green on /cc0 and home.'))return;
  fetch('/cc0done/{CATEGORY}',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{done:!on}})}})
    .then(r=>r.json()).then(j=>setDoneUI(j.done)).catch(e=>alert('Save failed: '+e));
}}
function exp(){{
  let scenes={json.dumps(export_scenes, ensure_ascii=False)};
  // replay scene merges (same as DOM: dst keeps ids, merged append)
  const byAnchor={{}};scenes.forEach(s=>byAnchor[s.anchor]=s);
  sceneMerges.forEach(mg=>{{
    const src=mg[0],dst=mg[1];
    const s=byAnchor[src];
    if(mg.length===3){{                      // cross-category: mark whole scene, keep content
      if(s&&!s._into&&!s._xinto)s._xinto={{category:mg[2],anchor_video:dst}};
      return;
    }}
    let d=byAnchor[dst];while(d&&d._into)d=byAnchor[d._into];
    if(!s||!d||s===d||s._into||s._xinto||d._xinto)return;
    let k=d.actors.length;
    s.actors.forEach(a=>{{a.label=lbl(k++);d.actors.push(a);}});
    d.videos.push(...s.videos);d.targets.push(...s.targets);d.no_face.push(...(s.no_face||[]));
    s._into=dst;
  }});
  // test set: propagate along merge chain; any member marked → whole scene test
  const finalOf=a=>{{let s=byAnchor[a];while(s&&s._into)s=byAnchor[s._into];return s;}};
  testSet.forEach(a=>{{const f=finalOf(a);if(f&&!f._xinto)f._test=true;}});
  // collect merged video groups for server scene_merges.json
  const groups=[];
  scenes.forEach(s=>{{if(!s._into&&sceneMerges.some(([a,b])=>{{let d=byAnchor[b];while(d&&d._into)d=byAnchor[d._into];return d===s;}}))groups.push(s.videos.slice());}});
  scenes=scenes.filter(s=>!s._into);
  scenes.forEach(s=>{{
    s.deleted_refs=[...(s.no_face||[])];delete s.no_face;   // faceless refs default deleted
    s.deleted_targets=[];
    s.actors.forEach(a=>{{
      s.deleted_refs.push(...a.refs.filter(fn=>delRefs.has(fn)));
      a.refs=a.refs.filter(fn=>!delRefs.has(fn));
      a.star=starMap[a.id]||null;
    }});
    const goneA=new Set(s.actors.filter(a=>!a.refs.length).map(a=>a.id));
    s.actors=s.actors.filter(a=>a.refs.length);
    s.deleted_targets=s.targets.filter(fn=>delTgts.has(fn));
    s.targets=s.targets.filter(fn=>!delTgts.has(fn));
    const b={{}};
    s.targets.forEach(fn=>{{
      const arr=(bindMap[fn]||[]).filter(aid=>!goneA.has(aid));
      if(arr.length)b[fn]=arr.map(aid=>s.actors.find(a=>a.id===aid)?.label||aid);
    }});
    s.bindings=b;
  }});
  scenes.forEach((s,i)=>{{s.scene=i;s.split=s._test?'test':'train';delete s._test;delete s._into;delete s.anchor;
    if(s._xinto){{s.merged_into=s._xinto;delete s._xinto;}}}});   // final assembly uses merged_into; split=test scenes benchmark only
  const blob=new Blob([JSON.stringify({{category:'{CATEGORY}',scene_merge_groups:groups,scenes:scenes}},null,1)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='cc0_scene_map_{CATEGORY}.json';a.click();
}}
sceneMerges=sceneMerges.filter(m=>{{                            // replay on load (drop stale in-category anchors)
  if(m.length===2)return applyMergeDOM(m[0],m[1]);
  const s=document.querySelector('.seq[data-anchor="'+m[0]+'"]');
  if(s)s.style.display='none';
  return true;   // legacy 3-tuple records auto-migrate to physical move below
}});
save();
render();
(function(){{                    // legacy cross-category records → auto physical move, then reload
  const legacy=sceneMerges.filter(m=>m.length===3);
  if(!legacy.length)return;
  (async()=>{{
    await fetch('/cc0state/{CATEGORY}',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:localStorage.getItem(K)}}).catch(()=>{{}});
    for(const m of legacy){{
      const r=await fetch('/cc0xmove',{{method:'POST',headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{src:CAT,anchor:m[0],dst:m[2],dstAnchor:m[1]}})}}).then(r=>r.json()).catch(e=>({{ok:false,err:String(e)}}));
      if(r.ok&&r.src_state)localStorage.setItem(K,JSON.stringify(r.src_state));
      else if(r.err!=='anchor-gone')return alert('Legacy cross-move auto-migrate failed: '+r.err);
    }}
    const cur=JSON.parse(localStorage.getItem(K));
    cur.smerge=(cur.smerge||[]).filter(m=>m.length===2);
    localStorage.setItem(K,JSON.stringify(cur));
    await fetch('/cc0state/{CATEGORY}',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:localStorage.getItem(K)}}).catch(()=>{{}});
    location.reload();
  }})();
}})();
</script></body></html>"""
OUT.write_text(html)
print(f"{CATEGORY}: {len(scenes)} scenes (multi-video {n_multi_scene}) {len(actors)} actors (split groups {n_split}) "
      f"{sum(len(s['targets']) for s in scenes)} targets -> {OUT}")
