import {createJointChart, initCaptureCharts, updateCaptureCharts, drawCaptureCharts} from './capture-chart.js';
const $ = id => document.getElementById(id);
const apiHeaders = {'Content-Type':'application/json','X-Console-Request':'1'};
let tasks = [], environments = [], workflow = null, capture = null, currentPage = 'tasks', toastTimer, previewBusy = false, execPreviewBusy = false;
let chartPoints = [], chartHover = -1;
const jointNames = ['J1','J2','J3','J4','J5','J6','夹爪'];
const pageMeta = {tasks:['WORKSPACE','任务中心'],capture:['STAGE 01','数据采集'],training:['STAGE 02','模型训练'],execution:['STAGE 03','任务执行']};
const captureLabels = {disconnected:'未连接',connecting:'连接中',ready:'设备就绪',recording:'正在录制',saving:'正在保存',error:'连接异常'};
const jobLabels = {idle:'尚未开始',starting:'正在初始化',running:'正在运行',stopping:'正在停止',stopped:'已停止',completed:'已完成',error:'运行失败'};

async function api(url, body) {
  const options = body === undefined ? {cache:'no-store'} : {method:'POST',headers:apiHeaders,body:JSON.stringify(body)};
  const response = await fetch(url, options);
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) throw Error(data?.error || `请求失败 (${response.status})`);
  return data;
}
function toast(message, error=false) {
  clearTimeout(toastTimer); $('toast').textContent=message; $('toast').className='toast'+(error?' error':''); $('toast').hidden=false;
  toastTimer=setTimeout(()=>$('toast').hidden=true,4500);
}
function setPill(el, state, text) {
  el.textContent=text; el.className='pill '+(state==='error'?'error':['running','starting','recording','connecting'].includes(state)?'busy':['ready','completed'].includes(state)?'good':state==='stopping'?'warn':'');
}
function route() {
  const wanted=(location.hash||'#tasks').slice(1); currentPage=pageMeta[wanted]?wanted:'tasks';
  document.querySelectorAll('.page').forEach(el=>el.hidden=el.id!==`page-${currentPage}`);
  document.querySelectorAll('[data-page]').forEach(el=>el.classList.toggle('active',el.dataset.page===currentPage));
  $('pageEyebrow').textContent=pageMeta[currentPage][0]; $('pageTitle').textContent=pageMeta[currentPage][1];
  if(currentPage==='training') requestAnimationFrame(drawTrainingChart);
  if(currentPage==='capture') requestAnimationFrame(drawCaptureCharts);
  if(currentPage==='execution') requestAnimationFrame(()=>executionChart.draw());
}
window.addEventListener('hashchange',route); route();

function selectedTask(){return tasks.find(t=>t.selected) || tasks[0];}
async function loadTasks() {
  tasks=await api('/api/tasks'); renderTasks();
}
function fillTaskPaths(task){
  for(const [formId,key] of [['trainingForm','dataset_dir'],['trainingForm','ckpt_dir'],['executionForm','ckpt_dir']]){
    const input=$(formId).elements[key];
    const taskId=task?.id||'';
    const value=task?.[key]||'';
    // Update untouched defaults, but preserve edits while this task remains selected.
    if(input.dataset.taskId!==taskId || input.value===(input.dataset.defaultPath||'')) input.value=value;
    input.dataset.taskId=taskId;
    input.dataset.defaultPath=value;
  }
}
function renderTasks() {
  const current=selectedTask();
  fillTaskPaths(current);
  if(!current){
    $('currentTaskName').textContent='暂无任务';$('taskEpisodes').textContent='0';$('taskModel').textContent='尚未训练';$('taskModelHint').textContent='请先创建任务';$('taskCount').textContent='0 个任务';
    $('trainDatasetPath').textContent=$('trainCkptPath').textContent='—';
    $('taskGrid').innerHTML=$('dialogTaskList').innerHTML='<p class="parameter-note">Dataset 下暂无任务文件夹，请新建任务。</p>';
    $('nextAction').textContent='先创建任务';$('nextActionHint').textContent='或将已有数据集文件夹放入 Dataset 后刷新。';$('nextActionLink').href='#tasks';$('nextActionLink').textContent='任务中心';
    $('deleteTask').disabled=true;$('renameTaskForm').hidden=true;return;
  }
  $('deleteTask').disabled=false;$('renameTaskForm').hidden=false;
  $('currentTaskName').textContent=current.name; $('taskEpisodes').textContent=current.episodes; $('taskModel').textContent=current.trained?'训练完成':'尚未训练'; $('taskModelHint').textContent=current.trained?'已找到最佳模型':'没有 policy_best.ckpt';
  $('trainDatasetPath').textContent=current.dataset_dir; $('trainCkptPath').textContent=current.ckpt_dir; $('taskCount').textContent=`${tasks.length} 个任务`;
  if(current.episodes===0){$('nextAction').textContent='先采集示教数据';$('nextActionHint').textContent='当前任务还没有 Episode。';$('nextActionLink').href='#capture';$('nextActionLink').textContent='进入采集';}
  else if(!current.trained){$('nextAction').textContent='开始训练模型';$('nextActionHint').textContent=`已有 ${current.episodes} 个 Episode 可用于训练。`;$('nextActionLink').href='#training';$('nextActionLink').textContent='进入训练';}
  else{$('nextAction').textContent='先运行仅推理';$('nextActionHint').textContent='模型已准备好，建议先检查实时输出。';$('nextActionLink').href='#execution';$('nextActionLink').textContent='进入执行';}
  const card=t=>`<button class="task-card ${t.selected?'selected':''}" data-task="${t.id}"><div class="task-card-head"><h3>${escapeHtml(t.name)}</h3>${t.selected?'<span class="pill good">当前</span>':''}</div><p>创建于 ${escapeHtml(t.created_at)}</p><code class="task-folder" title="${escapeHtml(t.dataset_dir)}">${escapeHtml(t.dataset_folder)}</code><div class="task-stats"><span>${t.episodes} Episodes</span><span>${t.trained?'模型已就绪':'未训练'}</span></div></button>`;
  $('taskGrid').innerHTML=tasks.map(card).join(''); $('dialogTaskList').innerHTML=tasks.map(t=>`<button class="dialog-task ${t.selected?'selected':''}" data-task="${t.id}"><span><strong>${escapeHtml(t.name)}</strong><small>${t.episodes} Episodes · ${t.trained?'模型已就绪':'未训练'}</small><small class="folder-line">文件夹：${escapeHtml(t.dataset_folder)}${t.folder_suffix?` · 唯一后缀：${escapeHtml(t.folder_suffix)}`:''}</small></span>${t.selected?'<span class="pill good">当前</span>':''}</button>`).join('');
  document.querySelectorAll('[data-task]').forEach(b=>b.onclick=()=>selectTask(b.dataset.task));
}
function escapeHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function renderEnvironment(){
  const select=$('runtimeEnvironment'),selected=environments.find(e=>e.selected);
  select.innerHTML=environments.map(e=>`<option value="${escapeHtml(e.name)}" ${e.selected?'selected':''}>${escapeHtml(e.name)} · ${e.execution_ready?'依赖完整':e.capture_ready?'可采集':e.training_ready?'可训练':'缺少依赖'}</option>`).join('');
  if(!selected){setPill($('environmentState'),'error','未找到环境');return;}
  const ready=selected.execution_ready;
  setPill($('environmentState'),ready?'ready':'error',ready?'依赖完整':'依赖不完整');
  $('environmentState').title=`Python: ${selected.python}\npiper_sdk: ${selected.modules.piper_sdk?'已安装':'缺少'}\npyorbbecsdk: ${selected.modules.pyorbbecsdk?'已安装':'缺少'}\npyrealsense2: ${selected.modules.pyrealsense2?'已安装':'缺少'}\npyAgxArm: ${selected.modules.pyAgxArm?'已安装':'缺少'}\ntorch: ${selected.modules.torch?'已安装':'缺少'}`;
}
async function loadEnvironments(){environments=await api('/api/environments');renderEnvironment();}
$('runtimeEnvironment').onchange=async e=>{const previous=environments.find(x=>x.selected)?.name;e.target.disabled=true;try{const selected=await api('/api/environments/select',{name:e.target.value});await loadEnvironments();toast(`已选择 Conda 环境“${selected.name}”`);}catch(err){toast(err.message,true);e.target.value=previous||'';}finally{e.target.disabled=false;}};
async function selectTask(id){
  if(selectedTask()?.id===id){$('taskDialog').close();return;}
  try{await api(`/api/tasks/${encodeURIComponent(id)}/select`,{});await loadTasks();$('taskDialog').close();toast('任务已切换');}
  catch(e){toast(e.message,true);}
}
async function openTasks(){await loadTasks();renderTasks();$('renameTaskForm').elements.name.value=selectedTask()?.name||'';$('taskDialog').showModal();}
$('taskSwitcher').onclick=$('openTaskModal').onclick=$('newTaskTop').onclick=openTasks; $('closeTaskDialog').onclick=()=>$('taskDialog').close();
$('taskDialog').onclick=e=>{if(e.target===$('taskDialog'))$('taskDialog').close();};
$('newTaskForm').onsubmit=async e=>{e.preventDefault();const name=new FormData(e.target).get('name');try{await api('/api/tasks',{name});e.target.reset();await loadTasks();$('taskDialog').close();toast(`已创建任务“${name}”`);}catch(err){toast(err.message,true);}};
$('renameTaskForm').onsubmit=async e=>{e.preventDefault();const task=selectedTask(),name=new FormData(e.target).get('name');if(!task)return;try{await api(`/api/tasks/${encodeURIComponent(task.id)}/rename`,{name});await loadTasks();$('renameTaskForm').elements.name.value=selectedTask().name;toast('任务、数据集和模型文件夹已同步重命名');}catch(err){toast(err.message,true);}};

initCaptureCharts($('jointCharts'));
const executionChart = createJointChart();
executionChart.init($('executionJointCharts'));
let executionChartSession = '', executionChartStamp = null;

function renderCapture(s){
  const previous=capture;capture=s;const online=['ready','recording','saving'].includes(s.state), active=s.worker_active;
  setPill($('captureState'),s.state,captureLabels[s.state]||s.state);setPill($('cameraBadge'),online?'ready':s.state,online?(s.config.mode==='demo'?'模拟信号':`反馈 ${Number(s.qpos_hz).toFixed(0)} Hz`):(captureLabels[s.state]||s.state));
  $('configFields').disabled=active;$('connect').disabled=active||!selectedTask();$('disconnect').disabled=!active;$('start').disabled=s.state!=='ready';$('stop').disabled=s.state!=='recording';
  if(!previous) for(const [k,v] of Object.entries(s.config)){const el=$('configForm').elements[k];if(el){if(el.type==='checkbox')el.checked=v;else el.value=v;}}
  $('count').textContent=Number(s.count).toLocaleString();$('elapsed').textContent=Number(s.elapsed).toFixed(1);$('age').textContent=s.image_age_ms==null?'—':Number(s.image_age_ms).toFixed(0);$('skipped').textContent=s.skipped;$('actionState').textContent=`ACTION ${s.action_state}`;$('datasetPath').textContent=s.dataset_dir;
  $('demoWatermark').hidden=!online||s.config.mode!=='demo';if(s.last_saved)$('saved').textContent=`已保存 ${s.last_saved}`;
  $('recordingTitle').textContent={recording:'正在记录示教',ready:'可以开始下一段示教',connecting:'等待设备就绪',error:'设备连接失败'}[s.state]||'准备采集';
  $('recordingHint').textContent=s.state==='recording'?'完成示教后点击停止并保存。关闭网页不会自动停止采集。':s.state==='ready'?'确认姿态和相机视角后，点击开始录制。':'连接后移动一次主臂，等待相机和从臂反馈就绪。';
  updateCaptureCharts(s);
  if(!online){$('liveImage').hidden=true;$('cameraEmpty').hidden=false;}
  if(s.error&&s.error!==previous?.error)toast(s.error,true);if(previous?.state==='recording'&&s.state==='ready')loadTasks().catch(()=>{});
}
function formData(form){const data=Object.fromEntries(new FormData(form));form.querySelectorAll('input[type=number]').forEach(i=>{if(i.value.trim()==='')delete data[i.name];else data[i.name]=Number(i.value);});form.querySelectorAll('input[type=checkbox]').forEach(i=>data[i.name]=i.checked);return data;}
async function command(url,data={}){try{return await api(url,data);}catch(e){toast(e.message,true);throw e;}}
$('configForm').onsubmit=async e=>{e.preventDefault();try{const data=formData(e.target);await syncPaths(data);renderCapture(await command('/api/connect',data));}catch{}};
$('start').onclick=()=>command('/api/start').then(renderCapture).catch(()=>{});$('stop').onclick=()=>command('/api/stop').then(s=>{renderCapture(s);loadTasks();toast('Episode 已保存');}).catch(()=>{});$('disconnect').onclick=()=>command('/api/disconnect').then(renderCapture).catch(()=>{});
$('liveImage').onload=()=>{$('liveResolution').textContent=`WRIST + EXTERNAL · ${$('liveImage').naturalWidth} × ${$('liveImage').naturalHeight}`;};

function renderTraining(job){
  setPill($('trainingState'),job.state,jobLabels[job.state]||job.state);const active=['starting','running','stopping'].includes(job.state);$('startTraining').disabled=active||!selectedTask();$('stopTraining').disabled=!['starting','running'].includes(job.state);$('trainingForm').querySelectorAll('input,select').forEach(i=>{if(i.id!=='stopTraining')i.disabled=active;});
  const m=job.latest||{}, train=m.train||{};$('epochMetric').textContent=m.epoch?`${m.epoch} / ${m.total_epochs}`:'0 / 0';$('lossMetric').textContent=formatMetric(train.loss);$('l1Metric').textContent=formatMetric(train.l1);$('klMetric').textContent=formatMetric(train.kl);
  const pct=m.total_epochs?Math.min(100,m.epoch/m.total_epochs*100):0;$('trainingProgress').style.width=`${pct}%`;
  $('trainingLog').textContent=job.logs?.length?job.logs.join('\n'):(job.error||'等待开始训练。');$('trainingLog').scrollTop=$('trainingLog').scrollHeight;
  chartPoints=job.metrics||[];$('chartEmpty').hidden=chartPoints.length>0;drawTrainingChart();
  if(job.state==='error'&&job.error) setPill($('trainingState'),'error','训练失败');
}
function formatMetric(v){return Number.isFinite(v)?v.toFixed(v<.01?6:4):'—';}
$('trainingForm').onsubmit=async e=>{e.preventDefault();try{await syncPaths(formData(e.target));await command('/api/training/start',formData(e.target));toast('训练已启动，初始化可能需要一些时间');}catch{}};
$('stopTraining').onclick=()=>command('/api/training/stop').then(()=>toast('正在停止训练')).catch(()=>{});
$('toggleTrainLog').onclick=()=>{const p=$('trainingLog');p.classList.toggle('collapsed');$('toggleTrainLog').textContent=p.classList.contains('collapsed')?'展开':'收起';};

function drawTrainingChart(){
  const canvas=$('trainingChart'),wrap=canvas.parentElement,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;const dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);if(!chartPoints.length)return;
  const series=[['loss','#3275b8'],['l1','#16816d'],['kl','#c67a27']];const vals=[];chartPoints.forEach(p=>series.forEach(([k])=>{const v=p.train?.[k];if(Number.isFinite(v))vals.push(v);}));if(!vals.length)return;
  let low=Math.min(...vals),high=Math.max(...vals);if(low===high){low*=.9;high*=1.1;}const pad=(high-low)*.08;low-=pad;high+=pad;const left=52,right=w-18,top=18,bottom=h-32,x=i=>left+(right-left)*(chartPoints.length===1?0:i/(chartPoints.length-1)),y=v=>bottom-(v-low)/(high-low)*(bottom-top);
  ctx.font='11px Segoe UI';ctx.lineWidth=1;for(let i=0;i<=4;i++){const yy=top+(bottom-top)*i/4,v=high-(high-low)*i/4;ctx.strokeStyle='#e3eaed';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(right,yy);ctx.stroke();ctx.fillStyle='#7b8b95';ctx.textAlign='right';ctx.fillText(v.toFixed(v<.1?4:2),left-8,yy+4);}
  ctx.textAlign='center';for(let i=0;i<=4;i++){const idx=Math.round((chartPoints.length-1)*i/4);ctx.fillStyle='#7b8b95';ctx.fillText(chartPoints[idx]?.epoch||1,x(idx),h-10);}
  series.forEach(([key,color])=>{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let started=false;chartPoints.forEach((p,i)=>{const v=p.train?.[key];if(!Number.isFinite(v))return;if(started)ctx.lineTo(x(i),y(v));else{ctx.moveTo(x(i),y(v));started=true;}});ctx.stroke();});
  if(chartHover>=0&&chartHover<chartPoints.length){const xx=x(chartHover);ctx.strokeStyle='#81929d';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(xx,top);ctx.lineTo(xx,bottom);ctx.stroke();ctx.setLineDash([]);}
}
$('trainingChart').addEventListener('mousemove',e=>{if(!chartPoints.length)return;const r=e.currentTarget.getBoundingClientRect(),left=52,right=r.width-18;chartHover=Math.max(0,Math.min(chartPoints.length-1,Math.round((e.clientX-r.left-left)/(right-left)*(chartPoints.length-1))));const p=chartPoints[chartHover],tip=$('chartTooltip');tip.innerHTML=`<strong>Epoch ${p.epoch}</strong><br>Loss ${formatMetric(p.train?.loss)}<br>L1 ${formatMetric(p.train?.l1)}<br>KL ${formatMetric(p.train?.kl)}`;tip.style.left=`${Math.min(r.width-120,Math.max(8,e.clientX-r.left+12))}px`;tip.style.top=`${Math.max(8,e.clientY-r.top-45)}px`;tip.hidden=false;drawTrainingChart();});
$('trainingChart').addEventListener('mouseleave',()=>{chartHover=-1;$('chartTooltip').hidden=true;drawTrainingChart();});new ResizeObserver(drawTrainingChart).observe($('trainingChart'));

function renderExecution(job){
  const live = ['starting','running','stopping'].includes(job.state);
  const runtime = job.latest || {};
  const session = live ? `${job.task_id}:${job.pid}` : executionChartSession;
  if (session !== executionChartSession) {
    executionChartSession = session; executionChartStamp = null;
    executionChart.update({dataset_dir:session,state:'connecting',worker_active:false});
  }
  if (live && Number.isFinite(runtime.sample_time) && runtime.sample_time !== executionChartStamp && Array.isArray(runtime.qpos)) {
    executionChartStamp = runtime.sample_time;
    executionChart.update({dataset_dir:session,state:'ready',worker_active:true,qpos_hz:1,qpos:runtime.qpos,action:runtime.command || []}, runtime.sample_time);
  }

  setPill($('executionState'),job.state,jobLabels[job.state]||job.state);const active=['starting','running','stopping'].includes(job.state);$('startExecution').disabled=active||!selectedTask();$('stopExecution').disabled=!['starting','running'].includes(job.state);$('executionForm').querySelectorAll('input,select').forEach(i=>i.disabled=active);
  const m=job.latest||{};$('wristAge').textContent=formatPlain(m.wrist_age_ms);$('externalAge').textContent=formatPlain(m.external_age_ms);$('syncDelta').textContent=formatPlain(m.sync_delta_ms);$('inferMs').textContent=formatPlain(m.inference_ms);$('inferHz').textContent=formatPlain(m.inference_hz);$('servoHz').textContent=formatPlain(m.servo_hz);$('execAge').textContent=formatPlain(m.image_age_ms ?? (Number.isFinite(m.wrist_age_ms)&&Number.isFinite(m.external_age_ms)?Math.max(m.wrist_age_ms,m.external_age_ms):undefined));$('executionStep').textContent=m.step==null?'STEP —':`STEP ${m.step}`;$('executionModeLabel').textContent=m.mode||($('executionForm').elements.controlMode.value==='control'?'CONTROL':'DRY-RUN');
  $('executionLog').textContent=job.logs?.length?job.logs.join('\n'):(job.error||'等待启动执行。');$('executionLog').scrollTop=$('executionLog').scrollHeight;setPill($('executionCameraBadge'),job.has_preview?'ready':job.state,job.has_preview?'实时画面':active?'正在初始化':'等待启动');
  if(!job.has_preview&&!active){$('executionImage').hidden=true;$('executionCameraEmpty').hidden=false;}
}
function formatPlain(v){return Number.isFinite(v)?Number(v).toFixed(1):'—';}
function updateExecutionMode(){const control=$('executionForm').elements.controlMode.value==='control';$('safetyBox').hidden=!control;$('startExecution').textContent=control?'确认并启动真实控制':'启动仅推理';if(!control)$('motionConfirmed').checked=false;}
document.querySelectorAll('[name=controlMode]').forEach(r=>r.onchange=updateExecutionMode);
$('executionForm').onsubmit=async e=>{e.preventDefault();const data=formData(e.target),control=e.target.elements.controlMode.value==='control';data.control=control;data.motion_confirmed=$('motionConfirmed').checked;if(control&&!data.motion_confirmed){toast('请先完成真实控制安全确认',true);return;}try{await syncPaths(data);await command('/api/execution/start',data);toast(control?'真实控制正在初始化，请保持急停可用':'仅推理模式正在初始化');}catch{}};
$('stopExecution').onclick=()=>command('/api/execution/stop').then(()=>toast('停止请求已发送，机械臂将保持当前位置')).catch(()=>{});

async function poll(){
  try{const [s,w]=await Promise.all([api('/api/status'),api('/api/workflow')]);renderCapture(s);workflow=w;renderTraining(w.training);renderExecution(w.execution);$('globalDot').classList.remove('bad');$('globalState').textContent='本地服务已连接';
    const active=s.state==='recording'?'正在采集':['starting','running','stopping'].includes(w.training.state)?'正在训练':['starting','running','stopping'].includes(w.execution.state)?'正在执行':'空闲';setPill($('phasePill'),active==='空闲'?'idle':'running',active);
    $('runtimeEnvironment').disabled=active!=='空闲';
  }catch(e){$('globalDot').classList.add('bad');$('globalState').textContent='本地服务不可用';}
  setTimeout(poll,700);
}
async function previewLoop(){
  if(!previewBusy&&currentPage==='capture'&&['ready','recording'].includes(capture?.state)){previewBusy=true;try{const r=await fetch('/api/preview.jpg',{cache:'no-store'});if(r.ok&&r.status!==204){const blob=await r.blob(),url=URL.createObjectURL(blob),img=$('liveImage'),old=img.dataset.url;img.src=url;img.dataset.url=url;if(old)URL.revokeObjectURL(old);img.hidden=false;$('cameraEmpty').hidden=true;}}catch{}finally{previewBusy=false;}}
  if(!execPreviewBusy&&currentPage==='execution'&&workflow?.execution?.has_preview){execPreviewBusy=true;try{const r=await fetch('/api/execution/preview.jpg',{cache:'no-store'});if(r.ok&&r.status!==204){const blob=await r.blob(),url=URL.createObjectURL(blob),img=$('executionImage'),old=img.dataset.url;img.src=url;img.dataset.url=url;if(old)URL.revokeObjectURL(old);img.hidden=false;$('executionCameraEmpty').hidden=true;}}catch{}finally{execPreviewBusy=false;}}
  setTimeout(previewLoop,120);
}
loadTasks().catch(e=>toast(e.message,true));loadEnvironments().catch(e=>{setPill($('environmentState'),'error','检测失败');toast(e.message,true);});poll();previewLoop();

async function syncPaths(data){
  const paths={};for(const key of ['dataset_dir','ckpt_dir'])if(data[key]?.trim())paths[key]=data[key].trim();
  if(!Object.keys(paths).length)return;
  await command(`/api/tasks/${encodeURIComponent(selectedTask().id)}/paths`,paths);
  // Keep submitted paths visible; loadTasks refreshes untouched defaults.
  await loadTasks();
}

let deletePlan=null;
$('refreshTasks').onclick=()=>loadTasks().then(()=>toast('已同步 Dataset 文件夹')).catch(e=>toast(e.message,true));
$('deleteTask').onclick=async()=>{
  const task=selectedTask();if(!task)return;
  try{
    deletePlan=await api(`/api/tasks/${encodeURIComponent(task.id)}/delete-plan`);
    $('deleteName').textContent=deletePlan.name;
    $('deletePaths').textContent=deletePlan.paths.map(p=>`${p.kind==='dataset_dir'?'数据集':'模型'}：${p.path}${p.exists?'':'（不存在，无需删除）'}`).join('\n');
    $('deleteConfirmation').value='';$('confirmDelete').disabled=true;
    $('taskDialog').close();$('deleteDialog').showModal();
  }catch(e){toast(e.message,true);}
};
$('deleteConfirmation').oninput=()=>{$('confirmDelete').disabled=$('deleteConfirmation').value!==deletePlan?.name;};
$('cancelDelete').onclick=()=>{$('deleteDialog').close();deletePlan=null;};
$('deleteForm').onsubmit=async e=>{
  e.preventDefault();if(!deletePlan)return;$('confirmDelete').disabled=true;
  try{
    const result=await api(`/api/tasks/${encodeURIComponent(deletePlan.id)}/delete`,{confirmation:$('deleteConfirmation').value,token:deletePlan.token});
    $('deleteDialog').close();deletePlan=null;await loadTasks();
    toast(result.warnings.length?result.warnings.join('\n'):'任务及对应文件夹已删除',!!result.warnings.length);
  }catch(err){toast(err.message,true);$('confirmDelete').disabled=false;}
};
