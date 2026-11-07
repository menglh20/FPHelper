import {PHASES,LABELS,TIME_KEYS,ISSUE_LABELS,SCORE_LABELS,clone,formatTime,parseTime,actionErrors,completionErrors,reviewedCount,scoreSummary,applyTime,filteredVideos} from './core.js';

const $=id=>document.getElementById(id);
const state={config:null,videos:[],record:null,meta:null,phase:0,revision:0,dirty:false,generation:0,
  promise:null,timer:null,undo:[],loaded:false,loading:false,loop:false,preview:null,navigation:0,conflict:false};
const player=$('video');
const COLORS=['#71859a','#468baf','#3c9b8f','#8775b4','#bc9455','#b47591'];
const STATUS={unannotated:'未标注',draft:'草稿',needs_review:'待复核',complete:'已完成'};
let session=sessionStorage.getItem('fp-annotator-session');
if(!session){session=crypto.randomUUID();sessionStorage.setItem('fp-annotator-session',session);}
let toastTimer;

function element(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function button(text,handler,className='button secondary'){const node=element('button',text,className);node.addEventListener('click',handler);return node;}
function toast(message,error=false){
  let target=$('toast');
  if($('dialog').open){target=$('dialog-feedback');if(!target){target=element('div');target.id='dialog-feedback';target.setAttribute('role','status');$('dialog').append(target);}target.className='dialog-feedback'+(error?' error':'');}
  else target.className='toast'+(error?' error':'');
  target.textContent=message;target.hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>target.hidden=true,error?7000:3500);
}
function saveStatus(text,error=false){$('save-status').textContent=text;$('save-status').className='save-status'+(error?' error':state.dirty?' saving':'');}
async function api(path,body){
  const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':state.config.csrf_token},body:JSON.stringify(body)};
  const response=await fetch(path,options);
  let value;
  try{value=await response.json();}catch{throw new Error('本地服务未返回有效数据，请确认服务仍在运行');}
  if(!response.ok){const error=new Error(value.error||'本地请求失败');error.status=response.status;throw error;}
  return value;
}
function draftPrefix(id=state.record?.video_id){return `fp-annotator:draft:${state.config.dataset_id}:${id}:`;}
function draftKey(){return draftPrefix()+session;}
function backupDraft(){
  if(!state.record)return;
  try{localStorage.setItem(draftKey(),JSON.stringify({base_revision:state.revision,record:state.record,saved_at:Date.now()}));}
  catch{saveStatus('浏览器草稿空间不足',true);}
}
function clearOwnDraft(){if(state.record)localStorage.removeItem(draftKey());}
function mutate(change,{render=false,preserveReview=false}={}){
  if(!state.record||state.loading)return;
  state.undo.push(clone(state.record));if(state.undo.length>80)state.undo.shift();
  if(typeof change==='function')change(state.record);else state.record=change;
  if(!preserveReview)state.record.status='draft';
  state.generation++;state.dirty=true;backupDraft();
  saveStatus('正在保存…');updateCurrentListItem();renderTimeline();renderSummary();
  if(render)renderEditor();
  clearTimeout(state.timer);state.timer=setTimeout(()=>flushSave(),500);
}
async function saveLoop(){
  if(state.conflict)return false;
  while(state.dirty){
    const generation=state.generation,record=clone(state.record);
    try{
      const {record:saved}=await api(`/api/videos/${record.video_id}`,{record,expected_revision:state.revision});
      state.revision=saved.revision;state.record.revision=saved.revision;
      state.record.updated_at=saved.updated_at;state.record.created_at=saved.created_at;
      if(state.generation===generation){state.dirty=false;clearOwnDraft();saveStatus(`已保存 · v${saved.revision}`);}
      else backupDraft();
      updateCurrentListItem();
    }catch(error){
      backupDraft();saveStatus('保存失败 · 草稿已保留',true);toast(error.message,true);
      if(error.status===409){state.conflict=true;showConflict(error.message);}
      return false;
    }
  }
  return true;
}
async function flushSave(){
  clearTimeout(state.timer);
  if(state.promise)return state.promise;
  if(!state.dirty)return true;
  state.promise=saveLoop();
  try{return await state.promise;}finally{state.promise=null;}
}
function showDialog(title,nodes){$('toast').hidden=true;$('dialog-feedback')?.remove();$('dialog-title').textContent=title;$('dialog-body').replaceChildren(...nodes);if(!$('dialog').open)$('dialog').showModal();}
function showConflict(message){
  showDialog('保存版本冲突',[element('p',message),button('导出本页草稿',downloadDraft),button('重新载入服务器版本',async()=>{
    if(!confirm('本页未保存修改会退出编辑；服务器历史版本和浏览器草稿仍保留。是否载入服务器版本？'))return;
    state.dirty=false;state.conflict=false;$('dialog').close();await selectVideo(state.record.video_id,true);
  })]);
}
function downloadBlob(blob,name){const url=URL.createObjectURL(blob),link=element('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function downloadDraft(){
  if(!state.record)return;
  downloadBlob(new Blob([JSON.stringify({schema_version:1,format:'fphelper-video-annotations',records:[state.record]},null,2)],{type:'application/json'}),`${state.record.video_id}-draft.json`);
}
function findDraft(){
  const drafts=[];
  for(let i=0;i<localStorage.length;i++){
    const key=localStorage.key(i);if(!key.startsWith(draftPrefix()))continue;
    try{const value=JSON.parse(localStorage.getItem(key));if(value.record&&value.record.video_id===state.record.video_id)drafts.push(value);}catch{}
  }
  return drafts.sort((a,b)=>b.saved_at-a.saved_at)[0];
}
function renderDraftBanner(){
  const banner=$('draft-banner'),draft=findDraft();banner.replaceChildren();banner.hidden=!draft;
  if(!draft)return;
  const compatible=draft.base_revision===state.revision&&draft.record.video_sha256===state.meta.video_sha256;
  banner.append(element('span',compatible?'发现本地未保存草稿。':'发现旧版本或其他页面的草稿，已保留供下载核对。'));
  if(compatible)banner.append(button('恢复草稿',()=>{mutate(clone(draft.record),{render:true,preserveReview:true});banner.hidden=true;}));
  banner.append(button('下载草稿',()=>downloadBlob(new Blob([JSON.stringify({schema_version:1,format:'fphelper-video-annotations',records:[draft.record]},null,2)],{type:'application/json'}),`${state.record.video_id}-recovered.json`)));
  banner.append(button('忽略提示',()=>banner.hidden=true));
}
function updateCurrentListItem(){
  if(state.record){const row=state.videos.find(v=>v.video_id===state.record.video_id);if(row)Object.assign(row,{status:state.record.status,patient_group_id:state.record.patient_group_id,reviewed_count:reviewedCount(state.record),revision:state.revision});}
  renderList();renderSummary();
}
function renderList(){
  const rows=filteredVideos(state.videos,$('search').value,$('filter').value),list=$('video-list');list.replaceChildren();
  for(const row of rows){
  const node=button('',()=>selectVideo(row.video_id),'video-item'+(row.video_id===state.record?.video_id?' selected':''));node.dataset.videoId=row.video_id;
    const top=element('div',undefined,'video-item-top');top.append(element('span',row.file_name,'file-label'),element('i',undefined,'status-dot '+row.status));
    const bottom=element('div',undefined,'item-bottom');bottom.append(element('span',row.patient_group_id||'未分组'),element('span',`${row.reviewed_count}/6 · ${STATUS[row.status]||'草稿'}`));
    node.append(top,element('div',row.video_id,'item-id'),bottom);const item=element('div');item.setAttribute('role','listitem');item.append(node);list.append(item);
  }
  if(!rows.length)list.append(element('div','没有符合条件的视频','empty-list'));
  const complete=state.videos.filter(v=>v.status==='complete').length,total=state.videos.length;
  $('video-count').textContent=total;$('dataset-progress').textContent=`${complete} / ${total} 已完成`;$('progress-percent').textContent=`${Math.round(complete/Math.max(total,1)*100)}%`;
  $('progress-bar').style.width=`${complete/Math.max(total,1)*100}%`;
}
function renderSummary(){
  if(!state.record)return;
  const status=state.revision===0&&!state.dirty?'unannotated':state.record.status;
  $('record-status').textContent=STATUS[status];$('record-status').className='status-badge '+status;
  const action=state.record.actions[PHASES[state.phase]];
  $('confirm-action').textContent=action.reviewed?'✓ 本动作已确认':'确认本动作';
  $('legacy-verified').checked=action.legacy_window_verified;$('legacy-verified').disabled=!action.reviewed||action.presence!=='present';
  $('action-errors').textContent=action.presence==='unreviewed'?'':actionErrors(action).join('；');
  $('action-count').textContent=`${reviewedCount(state.record)} / 6`;
  const index=state.videos.findIndex(v=>v.video_id===state.record.video_id);
  $('position').textContent=`${index+1} / ${state.videos.length}`;
  $('previous-video').disabled=index<=0;$('next-video').disabled=index>=state.videos.length-1;$('undo').disabled=!state.undo.length;
  const summary=scoreSummary(state.record);$('score-filled').textContent=`${summary.filled} / 13`;$('score-total').textContent=summary.total??'—';$('score-reason').textContent=summary.reason;
  const enabled=state.record.independent_scores.enabled;
  $('rater-id').disabled=!enabled;
  for(const select of document.querySelectorAll('[data-score]'))select.disabled=!enabled||!state.record.independent_scores.rater_id.trim();
}
function updateAvailability(){
  const ready=Boolean(state.record)&&!state.loading;
  document.querySelector('.editor').inert=!ready;
  for(const id of ['load-video','play','step-back','step-forward','scrubber','history-button','review-record','complete-record'])$(id).disabled=!ready;
  if(ready)renderSummary();else for(const id of ['undo','previous-video','next-video'])$(id).disabled=true;
}
function renderTimeline(){
  const root=$('timeline');root.replaceChildren();if(!state.record)return;
  const duration=state.meta.duration_s;
  for(const [i,phase] of PHASES.entries()){
    const a=state.record.actions[phase],row=element('div',undefined,'timeline-row'+(i===state.phase?' selected':''));row.style.setProperty('--phase',COLORS[i]);
    const label=button('',()=>selectPhase(i),'phase-select');label.append(element('span',String(i+1).padStart(2,'0'),'phase-number'),element('span',LABELS[i]),element('span',a.reviewed?'✓':'','phase-check'));
    const track=element('div',undefined,'timeline-track');track.setAttribute('role','slider');track.setAttribute('aria-label',`${LABELS[i]}视频定位轨道`);track.setAttribute('aria-valuemin','0');track.setAttribute('aria-valuemax',String(duration));track.setAttribute('aria-valuenow',String(state.loaded?player.currentTime:0));track.tabIndex=0;
    if($('show-reference').checked){const ref=element('div',undefined,'timeline-reference');ref.style.left=`${i/6*100}%`;ref.style.width=`${100/6}%`;track.append(ref);}
    if(a.presence==='present'&&a.start_s!==null&&a.end_s!==null&&a.end_s>=a.start_s){
      const bar=element('div',undefined,'timeline-bar');bar.style.left=`${a.start_s/duration*100}%`;bar.style.width=`${(a.end_s-a.start_s)/duration*100}%`;track.append(bar);
      if(a.hold_start_s!==null&&a.hold_end_s!==null&&a.hold_end_s>=a.hold_start_s){const hold=element('div',undefined,'timeline-hold');hold.style.left=`${a.hold_start_s/duration*100}%`;hold.style.width=`${(a.hold_end_s-a.hold_start_s)/duration*100}%`;track.append(hold);}
    }
    if(a.presence==='present'&&a.peak_s!==null){const peak=element('div',undefined,'timeline-peak');peak.style.left=`${a.peak_s/duration*100}%`;track.append(peak);}
    const cursor=element('div',undefined,'playhead');cursor.style.left=`${(state.loaded?player.currentTime:0)/duration*100}%`;track.append(cursor);
    track.addEventListener('click',async event=>{selectPhase(i);const ratio=(event.clientX-track.getBoundingClientRect().left)/track.getBoundingClientRect().width;if(await ensureLoaded())seek(ratio*duration);});
    track.addEventListener('keydown',event=>{if(event.key==='Enter'){selectPhase(i);previewAction();}});
    const timing=a.presence==='missing'?'缺少动作':a.presence==='uncertain'?'无法判断':`${a.start_s===null?'—':a.start_s.toFixed(2)} – ${a.end_s===null?'—':a.end_s.toFixed(2)} s`;
    row.append(label,track,element('span',timing,'timeline-times'));root.append(row);
  }
}
function renderQuality(id,flags,handler){
  const root=$(id);root.replaceChildren();
  for(const [key,label] of Object.entries(ISSUE_LABELS)){
    const chip=button(label,()=>handler(key),'quality-chip'+(flags.includes(key)?' active':''));chip.setAttribute('aria-pressed',String(flags.includes(key)));root.append(chip);
  }
}
function toggleFlag(target,key){const position=target.indexOf(key);if(position<0)target.push(key);else target.splice(position,1);}
function timeField(field,label,shortcut){
  const row=element('label',undefined,'time-field'),name=element('span',label);if(shortcut)name.append(element('span',shortcut,'shortcut'));
  const input=element('input');input.id=`time-${field}`;input.type='text';input.inputMode='decimal';input.placeholder='未标注';input.setAttribute('aria-label',label+'时间');
  input.addEventListener('input',()=>{try{const value=parseTime(input.value);input.removeAttribute('aria-invalid');setTime(field,value,false);}catch{input.setAttribute('aria-invalid','true');}});
  input.addEventListener('change',()=>{try{const value=parseTime(input.value);setTime(field,value);}catch(error){toast(error.message,true);input.value=state.record.actions[PHASES[state.phase]][field]===null?'':formatTime(state.record.actions[PHASES[state.phase]][field]);}input.removeAttribute('aria-invalid');});
  row.append(name,input,button('取当前',()=>markTime(field),'button secondary'));return row;
}
function createFields(){
  for(const [key,label,shortcut] of [['start_s','起始','S'],['peak_s','峰值','P'],['end_s','结束','E']])$('time-fields').append(timeField(key,label,shortcut));
  for(const [key,label,shortcut] of [['hold_start_s','保持起','['],['hold_end_s','保持止',']']])$('hold-fields').append(timeField(key,label,shortcut));
  let index=0;
  for(const [key,label] of Object.entries(SCORE_LABELS)){
    if([0,3,8].includes(index))$('score-fields').append(element('div',index===0?'静态对称性':index===3?'自主运动':'联动','score-group-title'));
    const row=element('label',undefined,'score-row'),select=element('select');select.dataset.score=key;select.setAttribute('aria-label',label+'评分');select.append(new Option('未评分',''));
    for(let value=state.config.score_limits[key][0];value<=state.config.score_limits[key][1];value++)select.append(new Option(String(value),String(value)));
    select.addEventListener('change',()=>mutate(r=>r.independent_scores.values[key]=select.value===''?null:Number(select.value)));
    row.append(element('span',label),select);$('score-fields').append(row);index++;
  }
}
function renderEditor(){
  if(!state.record)return;
  const a=state.record.actions[PHASES[state.phase]],r=state.record;
  $('phase-number').textContent=String(state.phase+1).padStart(2,'0');$('phase-title').textContent=LABELS[state.phase];$('presence').value=a.presence;
  for(const key of TIME_KEYS){$(`time-${key}`).value=a[key]===null?'':formatTime(a[key]);$(`time-${key}`).disabled=['missing','uncertain'].includes(a.presence);$(`time-${key}`).nextSibling.disabled=['missing','uncertain'].includes(a.presence);}
  $('action-notes').value=a.notes;$('legacy-verified').checked=a.legacy_window_verified;$('legacy-verified').disabled=!a.reviewed||a.presence!=='present';$('legacy-method').value=r.legacy_verification_method;
  $('confirm-action').textContent=a.reviewed?'✓ 本动作已确认':'确认本动作';$('action-errors').textContent=a.presence==='unreviewed'?'':actionErrors(a).join('；');
  for(const [id,key] of [['annotator-id','annotator_id'],['patient-group','patient_group_id'],['affected-side','affected_side'],['mirrored','mirrored'],['camera','camera'],['record-notes','notes']])$(id).value=r[key];
  $('source-side').textContent=state.meta.source_side&&state.meta.source_side!=='unknown'?`原清单患侧参考：${{left:'患者左侧',right:'患者右侧'}[state.meta.source_side]||state.meta.source_side}，仍需核实。`:'';
  $('scores-enabled').checked=r.independent_scores.enabled;$('rater-id').value=r.independent_scores.rater_id;$('score-notes').value=r.independent_scores.notes;
  for(const select of document.querySelectorAll('[data-score]'))select.value=r.independent_scores.values[select.dataset.score]??'';
  renderQuality('action-quality',a.quality_flags,key=>mutate(record=>{toggleFlag(record.actions[PHASES[state.phase]].quality_flags,key);record.actions[PHASES[state.phase]].reviewed=false;record.actions[PHASES[state.phase]].legacy_window_verified=false;},{render:true}));
  renderQuality('global-quality',r.quality_flags,key=>mutate(record=>toggleFlag(record.quality_flags,key),{render:true}));
  renderSummary();
}
function selectPhase(index){state.phase=Math.min(5,Math.max(0,index));state.preview=null;renderEditor();renderTimeline();}
async function selectVideo(id,reload=false){
  if(!reload&&id===state.record?.video_id)return;
  if(!await flushSave()){toast('当前草稿尚未保存，请先处理保存提示。',true);return;}
  const navigation=++state.navigation;state.loading=true;updateAvailability();saveStatus('正在读取视频信息…');
  try{
    const {record,meta}=await api(`/api/videos/${id}`);if(navigation!==state.navigation)return;
    state.mediaCancel?.();player.pause();player.removeAttribute('src');player.load();player.hidden=true;$('player-placeholder').hidden=false;
    state.record=record;state.meta=meta;state.revision=record.revision;state.dirty=false;state.conflict=false;state.generation=0;state.undo=[];state.loaded=false;state.preview=null;state.phase=0;
    $('current-name').textContent=meta.file_name;$('duration').textContent=formatTime(meta.duration_s);$('current-time').textContent=formatTime(0);$('scrubber').max=meta.duration_s;$('scrubber').value=0;$('load-video').disabled=false;
    $('video-meta').textContent=`${meta.width} × ${meta.height} · ${meta.codec.toUpperCase()} · ${meta.fps_hint.toFixed(2)} fps（平均）`;
    const preference=localStorage.getItem('fp-annotator-reviewer');if(preference&&!record.annotator_id&&record.revision===0)record.annotator_id=preference;
    renderEditor();renderTimeline();renderList();renderDraftBanner();saveStatus(record.revision?`已保存 · v${record.revision}`:'尚未修改');
    localStorage.setItem(`fp-annotator-last:${state.config.dataset_id}`,id);
  }catch(error){toast(error.message,true);saveStatus('读取失败',true);}finally{if(navigation===state.navigation){state.loading=false;updateAvailability();}}
}
async function ensureLoaded(){
  if(!state.record||state.loading)return false;
  if(state.loaded&&player.readyState>=1)return true;
  if(state.mediaPromise)return state.mediaPromise;
  const id=state.record.video_id;
  player.src=`/media/${state.record.video_id}`;player.hidden=false;$('player-placeholder').hidden=true;state.loaded=true;
  state.mediaPromise=new Promise(resolve=>{
    const finish=value=>{clearTimeout(timeout);player.removeEventListener('loadedmetadata',ready);player.removeEventListener('error',failed);state.mediaCancel=null;state.mediaPromise=null;resolve(value&&state.record?.video_id===id);};
    const ready=()=>finish(true),failed=()=>{toast('浏览器无法播放该文件。建议使用 H.264 编码的 MP4，或更换支持该格式的浏览器。',true);finish(false);};
    const timeout=setTimeout(()=>{toast('视频载入较慢，请确认文件已完整下载到本机。',true);finish(false);},15000);
    state.mediaCancel=()=>finish(false);
    player.addEventListener('loadedmetadata',ready,{once:true});player.addEventListener('error',failed,{once:true});
    player.load();
  });
  return state.mediaPromise;
}
function seek(value){if(!state.loaded)return;state.preview=null;player.currentTime=Math.max(0,Math.min(state.meta.duration_s,Number(value)));updatePlayback();}
async function togglePlay(){if(!await ensureLoaded())return;if(player.paused){try{await player.play();}catch{toast('无法播放视频，请检查浏览器解码支持。',true);}}else player.pause();}
function setTime(key,value,render=true){
  if(value!==null&&(!Number.isFinite(value)||value<0||value>state.meta.duration_s+0.001)){toast('时间超出视频范围',true);renderEditor();return;}
  if(key.startsWith('hold_'))$('hold-details').open=true;
  const current=state.record.actions[PHASES[state.phase]];
  if(current[key]===value&&current.presence==='present'){if(render)renderEditor();return;}
  mutate(applyTime(state.record,PHASES[state.phase],key,value),{render});
}
function markTime(key){if(!state.loaded||player.readyState<1){toast('请先载入视频，再取当前时间');return;}setTime(key,Math.round(Math.min(player.currentTime,state.meta.duration_s)*1000)/1000);}
async function previewAction(reference=false){
  if(!state.record)return;
  let start,end;
  if(reference){start=state.phase/6*state.meta.duration_s;end=(state.phase+1)/6*state.meta.duration_s;$('show-reference').checked=true;renderTimeline();}
  else{const a=state.record.actions[PHASES[state.phase]];start=a.start_s;end=a.end_s;if(start===null||end===null||start>=end){toast('请先设置有效的动作起止时间');return;}}
  if(!await ensureLoaded())return;
  player.currentTime=start;state.preview={start,end};try{await player.play();}catch{toast('点击播放以查看该区间');}
}
function updatePlayback(){
  const time=state.loaded?player.currentTime:0;$('current-time').textContent=formatTime(time);$('scrubber').value=time;
  for(const line of document.querySelectorAll('.playhead')){line.style.left=`${time/Math.max(state.meta?.duration_s||1,1)*100}%`;line.parentElement.setAttribute('aria-valuenow',String(time));}
  $('play').textContent=player.paused?'▶':'Ⅱ';
  let bounds=state.preview;
  if(!bounds&&state.loop&&state.record){const a=state.record.actions[PHASES[state.phase]];if(a.start_s!==null&&a.end_s!==null&&a.start_s<a.end_s)bounds={start:a.start_s,end:a.end_s};}
  if(bounds&&time>=bounds.end-0.015){if(state.loop){player.currentTime=bounds.start;}else{player.pause();state.preview=null;}}
}
function confirmAction(){
  if(!state.record)return;const a=state.record.actions[PHASES[state.phase]],errors=actionErrors(a);
  if(errors.length){toast(errors.join('\n'),true);return;}
  mutate(r=>r.actions[PHASES[state.phase]].reviewed=true,{render:true});
  if($('auto-next').checked&&state.phase<5)selectPhase(state.phase+1);
}
function undo(){
  if(!state.undo.length)return;const snapshot=state.undo.pop();state.record=snapshot;state.record.revision=state.revision;state.record.status='draft';state.generation++;state.dirty=true;backupDraft();renderEditor();renderTimeline();updateCurrentListItem();clearTimeout(state.timer);state.timer=setTimeout(()=>flushSave(),200);
}
async function nextVideo(direction){const index=state.videos.findIndex(v=>v.video_id===state.record?.video_id);const next=state.videos[index+direction];if(next)await selectVideo(next.video_id);}
function switchTab(name){for(const tab of document.querySelectorAll('.tab')){const active=tab.dataset.tab===name;tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));}for(const key of ['actions','metadata','scores'])$(`tab-${key}`).hidden=key!==name;}

async function download(kind){
  if(!await flushSave())return;
  try{const response=await fetch(`/api/export/${kind}`);if(!response.ok){const value=await response.json();throw new Error(value.error);}const name=response.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1]||'annotations.json';downloadBlob(await response.blob(),name);$('dialog').close();}catch(error){toast(error.message,true);}
}
function showExports(){
  const grid=element('div',undefined,'dialog-grid');
  for(const [kind,label] of [['annotations','全部已保存标注 JSON · 含动作区间、质量及独立评分'],['patient-map','完整患者分组 CSV · 所有视频需已填写患者组'],['patient-template','患者分组模板 CSV · 保留已填代号，空白待补'],['action-evidence','既有模型动作核验证据 ZIP · 仅含人工勾选的模型窗口']])grid.append(button(label,()=>download(kind)));
  grid.append(button('当前页面草稿 JSON · 便于保存失败时备份',downloadDraft));
  showDialog('导出本地标注',[element('p','实际动作区间与既有模型的等分窗口分别记录。JSON 可在本工具重新导入；患者分组和核验证据可供现有算法工具使用。'),grid]);
}
async function importFile(file){
  if(!file||!await flushSave())return;
  if(file.size>20*1024*1024){toast('导入文件不能超过 20 MB',true);return;}
  try{
    const content=await file.text();let request;
    if(file.name.toLowerCase().endsWith('.csv')){if(!confirm('导入分组表会更新其中非空的患者组，其他标注保留，旧版本可在历史恢复。是否继续？'))return;request={kind:'patient-map',csv:content};}
    else request={kind:'annotations',bundle:JSON.parse(content)};
    saveStatus('正在导入…');const result=await api('/api/import',request);
    state.videos=(await api('/api/videos')).videos;await selectVideo(state.record.video_id,true);
    toast(`已导入 ${result.imported} 条，跳过 ${result.skipped} 条。JSON 导入会保留服务器已有标注。`);
  }catch(error){toast(error.message,true);saveStatus('导入失败',true);}finally{$('import-file').value='';}
}
async function showHistory(){
  if(!state.record||!await flushSave())return;
  try{
    const {history}=await api(`/api/videos/${state.record.video_id}/history`),nodes=[element('p','恢复会创建一个新版本，已有历史保留。显示最近 100 个保存版本。')];
    for(const item of history){const row=element('div',undefined,'history-row');row.append(element('span',`v${item.revision} · ${item.annotator_id||'未署名'} · ${STATUS[item.status]}\n${new Date(item.updated_at).toLocaleString('zh-CN')}`));const restore=button(item.revision===state.revision?'当前版本':'恢复',async()=>{
      if(!confirm(`恢复到 v${item.revision} 的内容，并创建一个新版本？`))return;
      try{await api(`/api/videos/${state.record.video_id}/restore`,{revision:item.revision,expected_revision:state.revision});$('dialog').close();await selectVideo(state.record.video_id,true);toast('已恢复为新版本');}catch(error){toast(error.message,true);}
    },'button small secondary');restore.disabled=item.revision===state.revision;row.append(restore);nodes.push(row);}
    if(!history.length)nodes.push(element('p','还没有保存记录。'));showDialog('标注历史版本',nodes);
  }catch(error){toast(error.message,true);}
}
function showHelp(){
  const grid=element('div',undefined,'shortcut-grid');for(const [keys,label] of [['空格','播放／暂停'],['1–6','切换动作'],['S / P / E','起始／峰值／结束取当前时间'],['[ / ]','保持区间起止取当前时间'],['← / →','后退／前进 0.04 秒'],['Shift + ← / →','后退／前进 1 秒'],['R','预览当前动作区间'],['V','确认当前动作'],['⌘/Ctrl + S','立即保存'],['⌘/Ctrl + Z','撤销本页修改']])grid.append(element('kbd',keys),element('span',label));
  showDialog('使用说明与快捷键',[element('p','先选择视频并载入；判断每个动作是否出现，设置起始、峰值和结束，检查后确认。缺少动作或无法判断时填写原因。保持区间与独立评分均为可选。'),grid,element('p','输入框内优先保留文字编辑键。动作标注按秒保存，时间零点是有效值。时间步进不会把变帧率视频宣称为严格逐帧。'),element('p','所有修改自动写入本机；保存失败时浏览器会保留草稿。原视频及原医生 CSV 只读。')]);
}
function bind(){
  $('search').addEventListener('input',renderList);$('filter').addEventListener('change',renderList);
  $('load-video').addEventListener('click',ensureLoaded);$('play').addEventListener('click',togglePlay);
  $('step-back').addEventListener('click',()=>seek(player.currentTime-0.04));$('step-forward').addEventListener('click',()=>seek(player.currentTime+0.04));
  $('speed').addEventListener('change',()=>player.playbackRate=Number($('speed').value));$('scrubber').addEventListener('input',()=>seek($('scrubber').value));
  $('loop').addEventListener('click',()=>{state.loop=!state.loop;$('loop').setAttribute('aria-pressed',String(state.loop));});
  $('show-reference').addEventListener('change',renderTimeline);$('preview-action').addEventListener('click',()=>previewAction());$('preview-reference').addEventListener('click',()=>previewAction(true));
  $('previous-phase').addEventListener('click',()=>selectPhase(state.phase-1));$('next-phase').addEventListener('click',()=>selectPhase(state.phase+1));
  $('presence').addEventListener('change',()=>mutate(r=>{const a=r.actions[PHASES[state.phase]];a.presence=$('presence').value;a.reviewed=false;a.legacy_window_verified=false;if(['missing','uncertain'].includes(a.presence))for(const key of TIME_KEYS)a[key]=null;},{render:true}));
  $('action-notes').addEventListener('input',()=>mutate(r=>{const a=r.actions[PHASES[state.phase]];a.notes=$('action-notes').value;a.reviewed=false;a.legacy_window_verified=false;}));
  $('confirm-action').addEventListener('click',confirmAction);$('legacy-method').addEventListener('input',()=>mutate(r=>r.legacy_verification_method=$('legacy-method').value));
  $('legacy-verified').addEventListener('change',()=>{if($('legacy-verified').checked&&!state.record.legacy_verification_method.trim()){$('legacy-verified').checked=false;toast('先填写核验方法，再确认模型窗口');$('legacy-method').focus();return;}mutate(r=>r.actions[PHASES[state.phase]].legacy_window_verified=$('legacy-verified').checked);});
  for(const [id,key] of [['annotator-id','annotator_id'],['patient-group','patient_group_id'],['affected-side','affected_side'],['mirrored','mirrored'],['camera','camera'],['record-notes','notes']])$(id).addEventListener(['affected-side','mirrored','camera'].includes(id)?'change':'input',()=>{
    mutate(r=>r[key]=$(id).value);if(key==='annotator_id')localStorage.setItem('fp-annotator-reviewer',$(id).value);
  });
  $('scores-enabled').addEventListener('change',()=>{if(!$('scores-enabled').checked&&Object.values(state.record.independent_scores.values).some(v=>v!==null)&&!confirm('关闭会清空本条辅助评分，已保存历史仍保留。继续？')){$('scores-enabled').checked=true;return;}mutate(r=>{r.independent_scores.enabled=$('scores-enabled').checked;if(!r.independent_scores.enabled)for(const key of Object.keys(r.independent_scores.values))r.independent_scores.values[key]=null;},{render:true});});
  $('rater-id').addEventListener('input',()=>mutate(r=>r.independent_scores.rater_id=$('rater-id').value));$('score-notes').addEventListener('input',()=>mutate(r=>r.independent_scores.notes=$('score-notes').value));
  $('previous-video').addEventListener('click',()=>nextVideo(-1));$('next-video').addEventListener('click',()=>nextVideo(1));$('undo').addEventListener('click',undo);$('history-button').addEventListener('click',showHistory);
  $('review-record').addEventListener('click',()=>mutate(r=>r.status='needs_review',{preserveReview:true}));
  $('complete-record').addEventListener('click',async()=>{const errors=completionErrors(state.record);if(errors.length){toast(errors.slice(0,5).join('\n'),true);return;}mutate(r=>r.status='complete',{preserveReview:true});if(await flushSave())toast('本条标注已完成并保存。患者分组与独立评分可继续补充。');});
  for(const tab of document.querySelectorAll('.tab'))tab.addEventListener('click',()=>switchTab(tab.dataset.tab));
  $('help-button').addEventListener('click',showHelp);$('export-button').addEventListener('click',showExports);$('import-button').addEventListener('click',()=>$('import-file').click());$('import-file').addEventListener('change',()=>importFile($('import-file').files[0]));
  $('dialog-close').addEventListener('click',()=>$('dialog').close());$('data-info').addEventListener('click',()=>showDialog('本地数据与标注',[element('p',`视频目录：${state.config.data_directory}`),element('p',`自动保存：${state.config.annotation_directory}/annotations.sqlite3`),element('p',`当前视频 SHA-256：${state.meta.video_sha256}`),element('p','服务器只监听 127.0.0.1。视频从原目录只读播放，界面不使用外部字体、CDN 或云端接口；标注导出后请按医院要求保管。')]));
  player.addEventListener('timeupdate',updatePlayback);player.addEventListener('play',updatePlayback);player.addEventListener('pause',updatePlayback);player.addEventListener('ended',()=>{if(state.loop){const a=state.record.actions[PHASES[state.phase]];if(a.start_s!==null){player.currentTime=a.start_s;player.play().catch(()=>{});}}});
  document.addEventListener('keydown',event=>{
    const editing=event.target.closest('input,textarea,select,[contenteditable=true]');
    if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='s'){event.preventDefault();flushSave();return;}
    if(editing||$('dialog').open||!state.record||state.loading)return;
    if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='z'){event.preventDefault();undo();return;}
    if(event.metaKey||event.ctrlKey||event.altKey)return;
    if(event.key===' '){event.preventDefault();togglePlay();}
    else if(/^[1-6]$/.test(event.key)){event.preventDefault();selectPhase(Number(event.key)-1);}
    else if(['s','p','e','[',']'].includes(event.key.toLowerCase())){event.preventDefault();markTime({s:'start_s',p:'peak_s',e:'end_s','[':'hold_start_s',']':'hold_end_s'}[event.key.toLowerCase()]);}
    else if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();seek(player.currentTime+(event.key==='ArrowLeft'?-1:1)*(event.shiftKey?1:0.04));}
    else if(event.key.toLowerCase()==='r'){event.preventDefault();previewAction();}
    else if(event.key.toLowerCase()==='v'){event.preventDefault();confirmAction();}
  });
  window.addEventListener('beforeunload',event=>{if(state.dirty){backupDraft();event.preventDefault();event.returnValue='';}});
  document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden'&&state.dirty){backupDraft();flushSave();}});
  let last=-1;function tick(){if(state.loaded&&Math.abs(player.currentTime-last)>0.015){last=player.currentTime;updatePlayback();}requestAnimationFrame(tick);}requestAnimationFrame(tick);
}
async function start(){
  try{state.config=await api('/api/config');state.videos=(await api('/api/videos')).videos;$('dataset-label').textContent=`${state.config.dataset_label} · 六阶段视频资料`;
    const missing=state.config.manifest_missing||[],extra=state.config.manifest_extra||[];
    if(missing.length||extra.length){$('dataset-warning').hidden=false;$('dataset-warning').textContent=`目录与训练清单不一致：缺少 ${missing.length} 段，新增 ${extra.length} 段。可继续标注已有视频；导出完整训练分组表前，请补齐视频或选择更新后的清单。缺少标识：${missing.join('、')||'无'}`;}
    createFields();bind();updateAvailability();renderList();const last=localStorage.getItem(`fp-annotator-last:${state.config.dataset_id}`),first=state.videos.find(v=>v.video_id===last)||state.videos[0];if(first)await selectVideo(first.video_id);}
  catch(error){$('connection-error').hidden=false;$('connection-error').textContent=`无法连接本地标注服务：${error.message}。请在终端启动 tools/annotator/server.py。`;saveStatus('连接失败',true);}
}
start();
