export const PHASES = ['rest', 'forehead_wrinkle', 'eye_closure', 'smile', 'snarl', 'lip_pucker'];
export const LABELS = ['自然放松', '抬眉', '轻闭眼', '露齿微笑', '呲牙', '嘟嘴'];
export const TIME_KEYS = ['start_s', 'hold_start_s', 'peak_s', 'hold_end_s', 'end_s'];
export const ISSUE_LABELS = {blur:'模糊', occlusion:'遮挡', head_motion:'头部移动', lighting:'光照不佳', speech:'说话干扰', blink:'眨眼干扰', wrong_action:'动作不符', tracking:'关键点待核查', other:'其他'};
export const SCORE_LABELS = {rest_eye:'静态眼',rest_cheek:'静态鼻唇沟',rest_mouth:'静态嘴',vol_forehead_wrinkle:'抬眉自主运动',vol_eye_closure:'轻闭眼自主运动',vol_smile:'微笑自主运动',vol_snarl:'呲牙自主运动',vol_lip_pucker:'嘟嘴自主运动',syn_forehead_wrinkle:'抬眉联动',syn_eye_closure:'轻闭眼联动',syn_smile:'微笑联动',syn_snarl:'呲牙联动',syn_lip_pucker:'嘟嘴联动'};
export const clone = value => JSON.parse(JSON.stringify(value));
export function formatTime(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const ms = Math.round(Math.max(0,value)*1000);
  return `${String(Math.floor(ms/60000)).padStart(2,'0')}:${String(Math.floor(ms/1000)%60).padStart(2,'0')}.${String(ms%1000).padStart(3,'0')}`;
}
export function parseTime(raw) {
  const value=String(raw).trim();
  if (!value) return null;
  if (/^\d+(\.\d{1,6})?$/.test(value)) return Number(value);
  const match=value.match(/^(\d+):(\d{1,2}(?:\.\d{1,6})?)$/);
  if (!match || Number(match[2])>=60) throw new Error('请填写秒数或 mm:ss.sss，例如 12.500 或 00:12.500');
  return Number(match[1])*60+Number(match[2]);
}
export function actionErrors(a) {
  if(a.presence==='unreviewed') return ['请先判断动作是否出现'];
  if(['missing','uncertain'].includes(a.presence)) return a.notes.trim()?[]:['请填写缺失或无法判断的原因'];
  const errors=[];
  if([a.start_s,a.peak_s,a.end_s].some(v=>v===null)) errors.push('请设置起始、峰值／代表帧和结束');
  else if(!(a.start_s<=a.peak_s && a.peak_s<=a.end_s && a.start_s<a.end_s)) errors.push('需要 起始 ≤ 峰值 ≤ 结束，且区间长度大于零');
  if((a.hold_start_s===null)!==(a.hold_end_s===null)) errors.push('保持区间起止需要同时填写');
  else if(a.hold_start_s!==null && !(a.start_s!==null && a.end_s!==null && a.start_s<=a.hold_start_s && a.hold_start_s<=a.peak_s && a.peak_s<=a.hold_end_s && a.hold_end_s<=a.end_s)) errors.push('保持区间需要位于动作范围内并包含峰值');
  return errors;
}
export function reviewedCount(record) { return PHASES.filter(p=>record.actions[p].reviewed).length; }
export function completionErrors(record) {
  const errors=record.annotator_id.trim()?[]:['请在「基本信息」填写标注人代号'];
  for(const [i,p] of PHASES.entries()) {
    errors.push(...actionErrors(record.actions[p]).map(e=>`${LABELS[i]}：${e}`));
    if(!record.actions[p].reviewed) errors.push(`${LABELS[i]}：尚未确认`);
  }
  const starts=PHASES.map(p=>record.actions[p]).filter(a=>a.presence==='present').map(a=>a.start_s);
  if(starts.every(s=>s!==null) && starts.some((s,i)=>i&&s<starts[i-1])) errors.push('动作起始顺序需要复核');
  return errors;
}
export function scoreSummary(record) {
  const scores=record.independent_scores, entries=Object.entries(scores.values), filled=entries.filter(([,v])=>v!==null).length;
  const actionsReady=PHASES.every(p=>record.actions[p].presence==='present'&&record.actions[p].reviewed);
  if(!scores.enabled || filled!==13 || !actionsReady) return {filled, total:null, raw:null, reason:!actionsReady?'动作尚未全部确认':'评分尚未完整'};
  const sum=prefix=>entries.filter(([k])=>k.startsWith(prefix)).reduce((a,[,v])=>a+v,0);
  const raw=4*sum('vol_')-5*sum('rest_')-sum('syn_');
  return {filled, raw, total:raw>=0&&raw<=100?raw:null, reason:raw<0||raw>100?'分项组合超出总分范围，请复核':''};
}
export function applyTime(record, phase, field, value) {
  const next=clone(record), a=next.actions[phase];
  a[field]=value;
  a.presence='present'; a.reviewed=false; a.legacy_window_verified=false; next.status='draft';
  return next;
}
export function filteredVideos(videos, query, filter) {
  const q=query.trim().toLowerCase();
  return videos.filter(v=>(!q||`${v.file_name} ${v.video_id} ${v.patient_group_id}`.toLowerCase().includes(q)) &&
    (filter==='all'||(filter==='unassigned'?!v.patient_group_id:filter==='unannotated'?v.revision===0:v.status===filter)));
}
