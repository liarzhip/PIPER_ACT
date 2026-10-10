const $ = id => document.getElementById(id);
const names = ['J1','J2','J3','J4','J5','J6','夹爪'];
let state = null, busy = false, selected = null, selectedMode = 'hardware', episodeData = null;
let frame = 0, playing = false, playTimer, joint = 0, selectionVersion = 0, previewBusy = false, activeTab = 'capture';
const stateLabels = {disconnected:'未连接',connecting:'连接中',ready:'设备就绪',recording:'正在录制',saving:'正在保存',error:'连接异常'};
function notify(text) {$('message').textContent = text; $('message').hidden = !text;}
async function api(url, body) {
  const r = await fetch(url, body === undefined ? {cache:'no-store'} : {method:'POST',headers:{'Content-Type':'application/json','X-Console-Request':'1'},body:JSON.stringify(body)});
  const data = await r.json();
  if (!r.ok) throw Error(data.error || `请求失败 (${r.status})`);
  return data;
}
function endpoint(suffix='') {return `/api/episodes/${encodeURIComponent(selected)}${suffix}?mode=${selectedMode}`;}
function setTab(tab) {
  if(tab==='capture'){location.href='/#capture';return;}
  activeTab = tab;
  for (const t of ['capture','review']) $(t).hidden = t !== tab;
  document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  if (tab === 'review') {loadEpisodes(); requestAnimationFrame(drawChart);} else pause();
}
document.querySelectorAll('.nav').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
$('joints').innerHTML = names.map((n,i)=>`<div class="joint"><strong>${n}${i===6?' / m':''}</strong><span id="q${i}">—</span><span id="a${i}">—</span></div>`).join('');
names.forEach((n,i)=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>{joint=i;drawChart();};$('jointTabs').append(b);});
function renderStatus(s) {
  const previous = state; state = s;
  $('connection').textContent = stateLabels[s.state] || s.state;
  $('connection').className = 'badge '+(s.state==='recording'?'rec':s.state==='ready'?'good':s.state==='error'?'warn':'');
  const online = ['ready','recording','saving'].includes(s.state);
  $('configFields').disabled = busy || s.worker_active;
  $('connect').disabled = busy || s.worker_active;
  $('disconnect').disabled = busy || !s.worker_active;
  if(!previous && s.worker_active) for(const [k,v] of Object.entries(s.config)) {const el=$('configForm').elements[k];if(el){if(el.type==='checkbox')el.checked=v;else el.value=v;}}
  $('start').disabled = busy || s.state !== 'ready';
  $('stop').disabled = busy || s.state !== 'recording';
  $('cameraBadge').textContent = online ? (s.config.mode==='demo'?'模拟信号':`反馈 ${Number(s.qpos_hz).toFixed(0)} Hz`) : stateLabels[s.state];
  $('cameraBadge').className = 'badge '+(online?'good':'');
  $('count').textContent = s.count.toLocaleString(); $('elapsed').textContent = s.elapsed.toFixed(1);
  $('age').textContent = s.image_age_ms === null ? '—' : s.image_age_ms.toFixed(0);
  $('age').style.color = s.image_age_ms > s.config.max_image_age*1000 ? '#b65820' : '';
  $('skipped').textContent = s.skipped; $('actionState').textContent = `ACTION ${s.action_state}`;
  $('datasetPath').textContent = s.dataset_dir;
  if (s.last_saved) $('saved').textContent = `已保存 ${s.last_saved} · 可前往回放与验证`;
  $('demoWatermark').hidden = !online || s.config.mode !== 'demo';
  $('recordingTitle').textContent = {recording:'正在记录示教',ready:'可以开始下一段示教',connecting:'等待设备就绪',error:'设备连接失败'}[s.state] || '准备采集';
  $('recordingHint').textContent = s.state==='recording' ? '完成示教后点击停止并保存。关闭网页不会自动停止采集。' : s.state==='ready' ? '确认起始姿态和相机视角后，点击开始录制。' : '连接后移动一次主臂，等待相机与从臂反馈就绪。';
  for(let i=0;i<7;i++) { $('q'+i).textContent = Number.isFinite(s.qpos[i])?s.qpos[i].toFixed(4):'—'; $('a'+i).textContent=Number.isFinite(s.action[i])?s.action[i].toFixed(4):'—'; }
  if (!online) {$('liveImage').hidden=true;$('cameraEmpty').hidden=false;}
  if (s.error && s.error !== previous?.error) notify(s.error);
  if(previous?.state==='recording' && s.state==='ready') loadEpisodes();
}
async function poll() {
  try {renderStatus(await api('/api/status'));}
  catch(e) { $('connection').textContent='服务不可用'; ['start','stop','connect','disconnect'].forEach(id=>$(id).disabled=true); notify(`无法连接本地服务：${e.message}`); }
  setTimeout(poll,600);
}
async function command(url,data={}) {
  busy=true;if(state)renderStatus(state);notify('');
  try {renderStatus(await api(url,data));}
  catch(e) {notify(e.message);}
  finally {busy=false;if(state)renderStatus(state);}
}
$('configForm').onsubmit=e=>{e.preventDefault();const data=Object.fromEntries(new FormData(e.target));for(const k of ['fps','camera_width','camera_height','camera_fps','max_image_age','duration','startup_timeout'])data[k]=Number(data[k]);data.judge_flag=e.target.elements.judge_flag.checked;command('/api/connect',data);};
$('start').onclick=()=>command('/api/start'); $('stop').onclick=()=>command('/api/stop'); $('disconnect').onclick=()=>command('/api/disconnect');
async function preview() {
  if(!previewBusy && activeTab==='capture' && ['ready','recording'].includes(state?.state)) {
    previewBusy=true;
    try {const r=await fetch('/api/preview.jpg',{cache:'no-store'});if(r.status===200){const blob=await r.blob();const url=URL.createObjectURL(blob);const img=$('liveImage');const old=img.dataset.url;img.src=url;img.dataset.url=url;if(old)URL.revokeObjectURL(old);img.hidden=false;$('cameraEmpty').hidden=true;}}
    catch {} finally {previewBusy=false;}
  }
  setTimeout(preview,100);
}
$('liveImage').onload=()=>{$('liveResolution').textContent=`WRIST · ${$('liveImage').naturalWidth} × ${$('liveImage').naturalHeight}`;};
async function loadEpisodes() {
  const mode=$('reviewMode').value;
  try {
    const rows=await api('/api/episodes?mode='+mode);if(mode!==$('reviewMode').value)return;
    $('episodes').replaceChildren();
    if(!rows.length){const p=document.createElement('p');p.className='meta';p.textContent='暂无 Episode。完成采集并保存后会显示在这里。';$('episodes').append(p);}
    for(const r of rows){const b=document.createElement('button');b.className='episode-row'+(selected===r.name&&selectedMode===mode?' selected':'');b.textContent=r.name;const small=document.createElement('small');small.textContent=r.error?'文件异常 · 可运行验证':`${r.frames} 帧 · ${r.duration} s · ${r.size_mb} MB`;b.append(small);b.onclick=()=>selectEpisode(r.name,mode);$('episodes').append(b);}
  } catch(e){notify(e.message);}
}
$('refresh').onclick=loadEpisodes;
$('reviewMode').onchange=()=>{selectionVersion++;pause();selected=null;episodeData=null;$('episodeDetail').hidden=true;$('reviewEmpty').hidden=false;loadEpisodes();};
async function selectEpisode(name,mode) {
  const version=++selectionVersion;pause();selected=name;selectedMode=mode;episodeData=null;frame=0;
  $('episodeDetail').hidden=false;$('reviewEmpty').hidden=true;$('episodeName').textContent=name;$('report').textContent='等待运行验证。';$('plotLinks').replaceChildren();
  $('validationStatus').textContent='尚未验证';$('validationStatus').className='badge';$('download').href=endpoint('/download');
  $('playImage').removeAttribute('src');$('play').disabled=true;$('seek').disabled=true;$('prev').disabled=true;$('next').disabled=true;
  $('episodeMeta').textContent='正在读取数据…';drawChart();loadEpisodes();
  try {
    const d=await api(endpoint('/data'));if(version!==selectionVersion)return;episodeData=d;
    $('seek').max=Math.max(0,d.frames-1);$('seek').disabled=false;$('play').disabled=false;$('prev').disabled=false;$('next').disabled=false;
    $('playDemo').hidden=!d.sim;$('episodeMeta').textContent=`${d.frames} 帧 · ${d.fps} Hz · ${d.duration} s · ${d.size_mb} MB${d.sim?' · 模拟数据，不用于训练':''}`;
    if(d.recording_error)notify('此 Episode 采集中断：'+d.recording_error);
    showFrame(0);drawChart();
  }catch(e){if(version===selectionVersion){$('episodeMeta').textContent='数据结构异常，请运行验证查看详情。';notify(e.message);}}
}
function pause(){playing=false;clearTimeout(playTimer);$('play').textContent='播放';}
function showFrame(value){if(!episodeData)return;frame=Math.max(0,Math.min(episodeData.frames-1,value));$('seek').value=frame;$('frameLabel').textContent=`${frame+1} / ${episodeData.frames}`;$('playImage').src=endpoint('/frame/'+frame);}
let playbackStarted=0, playbackFrame=0;
function tick(){if(!playing||!episodeData)return;const target=playbackFrame+Math.floor((performance.now()-playbackStarted)/1000*episodeData.fps*Number($('speed').value));if(target>=episodeData.frames){showFrame(episodeData.frames-1);pause();return;}if(target!==frame)showFrame(target);playTimer=setTimeout(tick,Math.max(16,1000/episodeData.fps/Number($('speed').value)));}
$('play').onclick=()=>{if(playing){pause();return;}if(!episodeData)return;if(frame>=episodeData.frames-1)showFrame(0);playing=true;$('play').textContent='暂停';playbackStarted=performance.now();playbackFrame=frame;tick();};
$('seek').oninput=()=>{pause();showFrame(Number($('seek').value));};$('prev').onclick=()=>{pause();showFrame(frame-1);};$('next').onclick=()=>{pause();showFrame(frame+1);};$('speed').onchange=()=>{if(playing){playbackStarted=performance.now();playbackFrame=frame;}};
document.addEventListener('keydown',e=>{if(activeTab!=='review'||!episodeData||['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.key.toLowerCase()==='a')$('prev').click();if(e.key.toLowerCase()==='d')$('next').click();});
async function validate(plots){
  if(!selected)return;const version=selectionVersion, url=endpoint('/validate');
  $('validate').disabled=$('plots').disabled=true;$('validationStatus').textContent='验证中…';$('report').textContent='正在检查 Episode，请稍候…';$('plotLinks').replaceChildren();
  try{const r=await api(url,{plots,max_image_age:Number($('validationAge').value),max_camera_sync_delta:Number($('validationSync').value)});if(version!==selectionVersion)return;
    $('validationStatus').textContent={pass:'检查通过',review:'需复核',fail:'验证失败'}[r.status];$('validationStatus').className='badge '+(r.status==='pass'?'good':r.status==='review'?'warn':'rec');$('report').textContent=r.report;
    r.plots.forEach(p=>{const a=document.createElement('a');a.textContent=p.replace('_qpos_action.png','').toUpperCase()+' PNG ↓';a.href=endpoint('/plots/'+p);a.download=p;$('plotLinks').append(a);});
  }catch(e){if(version===selectionVersion){$('validationStatus').textContent='验证未完成';$('report').textContent=e.message;notify(e.message);}}
  finally{$('validate').disabled=$('plots').disabled=false;}
}
$('validate').onclick=()=>validate(false);$('plots').onclick=()=>validate(true);
function drawChart(){
  [...$('jointTabs').children].forEach((b,i)=>b.classList.toggle('active',i===joint));
  const canvas=$('chart'),width=canvas.clientWidth;if(!width)return;
  const height=240,dpr=window.devicePixelRatio||1;canvas.width=width*dpr;canvas.height=height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  if(!episodeData)return;
  const scale=joint===6?1000:180/Math.PI;
  const lines=[episodeData.qpos,episodeData.action].map(rows=>rows.map(r=>r[joint]===null?null:r[joint]*scale));
  const values=lines.flat().filter(Number.isFinite);if(!values.length)return;
  let low=Math.min(...values),high=Math.max(...values),pad=Math.max((high-low)*.15,.001);low-=pad;high+=pad;
  const left=60,right=width-24,top=15,bottom=198, times=episodeData.time;
  const validTimes=times.filter(Number.isFinite),maxT=Math.max(...validTimes,1e-3);
  const x=i=>left+(times[i]??0)/maxT*(right-left),y=v=>bottom-(v-low)/(high-low)*(bottom-top);
  ctx.font='11px Segoe UI';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const v=low+(high-low)*i/4,yy=y(v);ctx.strokeStyle='#e6ecef';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(right,yy);ctx.stroke();ctx.fillStyle='#788a96';ctx.textAlign='right';ctx.fillText(v.toFixed(2),left-10,yy+4);ctx.textAlign='center';const xx=left+(right-left)*i/4;ctx.fillText((maxT*i/4).toFixed(1)+' s',xx,bottom+24);}
  lines.forEach((line,k)=>{ctx.strokeStyle=k?'#c38a39':'#087f6e';ctx.lineWidth=1.7;ctx.beginPath();let pen=false;line.forEach((v,i)=>{if(v===null||!Number.isFinite(times[i])){pen=false;return;}if(pen)ctx.lineTo(x(i),y(v));else ctx.moveTo(x(i),y(v));pen=true;});ctx.stroke();});
}
new ResizeObserver(drawChart).observe($('chart'));
setTab('review');poll();preview();
