'use strict';
(() => {
  const rows=window.EXPERIMENT_CATALOG, heads=window.EXPERIMENT_HEADS;
  const byId=Object.fromEntries(rows.map(r=>[r.id,r]));
  const el=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const count=n=>n==null?'보관 기록 참고':n.toLocaleString('en-US');
  const data=r=>r.total==null?'과거 전체 membership 미확인':`실제 ${count(r.real)} + 합성 ${count(r.synthetic)} + 배경-only ${count(r.background)} = ${count(r.total)}장`;
  const base=id=>id.replace(/_seed[012]$/i,'');
  const dialog=document.createElement('dialog');dialog.id='experimentDialog';dialog.className='lookup-modal';dialog.setAttribute('aria-labelledby','lookupTitle');
  dialog.innerHTML=`<button class="lookup-close" id="lookupClose" aria-label="실험 사전 닫기">닫기 ×</button><h2 id="lookupTitle">실험 설정 찾아보기</h2><div class="lookup-search"><label>이름·데이터·설명 검색<input id="lookupSearch" type="search" placeholder="예: BG0, M2_seed0, AnyDoor" autocomplete="off"></label><label>실험 선택<select id="lookupSelect"></select></label></div><p class="lookup-search-note" id="lookupNote"></p><p id="lookupEmpty" class="lookup-empty">일치하는 실험이 없습니다. BG0, M2, H1_S처럼 입력해 보세요.</p><div id="lookupDetails" aria-live="polite"></div><p><a href="experiment_guide.html">전체 실험표·지표·색상 사전 열기</a> · <a href="experiment_guide.html#guide-metrics">지표 뜻 보기</a></p>`;
  document.body.append(dialog);
  function details(id){
    const r=byId[base(id)];
    if(!r){el('lookupDetails').innerHTML=heads[id]?`<h3>${esc(id)} · 모델 구조</h3><p>${esc(heads[id])}</p><p>데이터 조건까지 보려면 ${id}_R(실제만) 또는 ${id}_S(휴리스틱 추가)를 선택하세요. 기본 M/L/A/BG 모델은 H2입니다.</p>`:'';return;}
    const seed=id.match(/_seed([012])$/);
    const fields=[['합성 방식',r.strategy],['모델 구조',r.architecture],['학습 반복',`${r.epochs} epochs / 등록 seed: ${r.seeds}${seed?' · 지금 읽은 job은 seed'+seed[1]:''}`],['평가',r.evaluation],['비교 목적',r.compare],['주의',r.note]];
    el('lookupDetails').innerHTML=`<h3>${esc(id)} · ${esc(r.label)}</h3><p class="lookup-data">${esc(data(r))}</p><dl class="lookup-definition">${fields.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>${r.alias?`<p class="lookup-notice">별칭: ${esc(r.alias)}의 checkpoint 재사용. 추가 학습 모델이 아닙니다.</p>`:''}`;
  }
  function filter(preferred){
    const query=el('lookupSearch').value.trim(),q=base(query.toUpperCase());
    if(!preferred&&(byId[q]||heads[q])){const seed=query.match(/_seed([012])$/i);preferred=q+(seed?'_seed'+seed[1]:'');}
    const suggestion=/^G[01]$/.test(q),found=rows.filter(r=>suggestion?r.id==='B'+q:JSON.stringify(r).toUpperCase().includes(q));
    el('lookupSelect').replaceChildren(...found.map(r=>new Option(`${r.id} · ${r.label}`,r.id)));
    if(preferred&&found.some(r=>r.id===base(preferred)))el('lookupSelect').value=base(preferred);
    if(preferred&&heads[preferred]){el('lookupSelect').prepend(new Option(preferred+' · 모델 구조 공통 정의',preferred));el('lookupSelect').value=preferred;}
    el('lookupEmpty').style.display=found.length?'none':'block';
    el('lookupNote').textContent=suggestion?`${q}는 정식 ID가 아닙니다. 배경 실험을 찾으셨다면 B${q} 설정을 확인하세요.`:'현재 설정 기준. 1학기는 보관 기록의 불확실성을 별도 표시합니다.';
    details(preferred||el('lookupSelect').value);
  }
  function open(id='BG0'){
    el('lookupSearch').value='';filter(id);
    if(!dialog.open)dialog.showModal();
    el('lookupSearch').focus();
  }
  el('lookupClose').onclick=()=>dialog.close();
  dialog.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();dialog.close();}});
  el('lookupSearch').oninput=()=>filter();el('lookupSelect').onchange=()=>details(el('lookupSelect').value);
  dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  document.addEventListener('click',e=>{const b=e.target.closest('[data-experiment]');if(b){e.preventDefault();open(b.dataset.experiment);}});
  // Annotate prose and table cells, never file paths, code blocks, controls, or existing links.
  const pattern=/(?<![\w/])(?:BG[01]|M[0-3]|L[0-3]|A[0-5]|H[012](?:_[RS])?|S[34](?:\+HardMining)?)(?:_seed[012])?(?![\w/])/g;
  for(const root of document.querySelectorAll('#report,.lookup-content,.lookup-intro,.case-content')){
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);const texts=[];
    while(walker.nextNode()){const t=walker.currentNode;if(!t.parentElement.closest('a,button,script,style,code,pre,select,textarea'))texts.push(t);}
    for(const t of texts){const matches=[...t.data.matchAll(pattern)];if(!matches.length)continue;let last=0;const frag=document.createDocumentFragment();
      for(const m of matches){frag.append(t.data.slice(last,m.index));const b=document.createElement('button');b.className='exp-ref';b.type='button';b.dataset.experiment=m[0];b.textContent=m[0];b.title=`${m[0]}의 데이터·모델·학습 설정 보기`;b.setAttribute('aria-label',b.title);frag.append(b);last=m.index+m[0].length;}frag.append(t.data.slice(last));t.replaceWith(frag);
    }
  }
  function card(id){const r=byId[base(id)];return `<strong>${esc(r.id)} · ${esc(r.label)}</strong><p>${esc(data(r))}</p><p>H2 · 150 epochs · 화면은 seed0 / Validation-best</p><button class="exp-ref" data-experiment="${r.id}">전체 설정 보기</button>`;}
  function compare(left,right){
    const a=byId[base(left)],b=byId[base(right)],ids=[a.id,b.id].sort().join('/');
    if(a.id===b.id)return '같은 모델입니다. 비교할 다른 데이터 조건을 선택하세요.';
    const bg={'BG0/M0':'합성 없는 조건에서 배경-only 2,129장을 추가한 영향','BG1/M1':'휴리스틱 혼합 조건에서 배경-only 2,129장을 추가한 영향','BG0/BG1':'같은 배경-only가 있는 상태에서 휴리스틱 3,000장을 추가한 영향'};
    if(bg[ids])return `${a.id} ↔ ${b.id}: ${bg[ids]}을 봅니다.`;
    if(a.group===b.group&&['M','L'].includes(a.group))return `${a.id} ↔ ${b.id}: 실제 ${count(a.real)}장을 공유하고 합성 ${a.strategy} / ${b.strategy}를 비교합니다.`;
    return `${a.id} ↔ ${b.id}: 실제 장수·합성 전략·배경 포함 여부 중 여러 조건이 달라질 수 있습니다. 하나의 요소만 바꾼 ablation으로 해석하지 마세요.`;
  }
  window.refreshModelDetails=()=>{
    if(!el('leftModel'))return;
    for(const side of ['left','right','live'])el(side+'Details').innerHTML=card(el(side+'Model').value);
    el('comparisonPurpose').textContent=compare(el('leftModel').value,el('rightModel').value)+' 이 화면은 한 이미지의 seed0 예시이며, 전체 성능은 보고서의 3seed 평균을 보세요.';
    const mode=el('mode').value,legend=document.querySelector('.legend');
    const labels=mode==='difference'?[['#42be80','정답·예측 겹침'],['#ffad42','정답만: 빠진 영역'],['#ef669d','예측만: 과도한 영역']]:mode==='overlay'?[['#42e680','정답 윤곽'],['#39aef0','예측 영역']]:[];
    legend.innerHTML=labels.map(([color,label])=>`<span><i class="dot" style="background:${color}"></i>${label}</span>`).join('')||(mode==='plain'?'원본 이미지: 마스크를 표시하지 않습니다.':'');
  };
  if(el('leftModel')){
    for(const side of ['left','right','live']){
      const select=el(side+'Model');for(const opt of select.options){const r=byId[base(opt.value)];opt.textContent=`${r.id} · 실제${r.real}${r.synthetic?' + '+(r.id.endsWith('1')?'휴리스틱':r.id.endsWith('2')?'AnyDoor 무작위':'AnyDoor 난도')+r.synthetic:''}${r.background?' + 배경'+r.background:''}`;}
      const d=document.createElement('div');d.id=side+'Details';d.className='model-explainer';select.closest('label').insertAdjacentElement('afterend',d);
    }
    const pair=document.querySelector('.pair'),purpose=document.createElement('p');purpose.id='comparisonPurpose';purpose.className='compare-explainer';purpose.setAttribute('aria-live','polite');pair.before(purpose);
    el('liveModel').addEventListener('change',window.refreshModelDetails);window.refreshModelDetails();
  }
})();
