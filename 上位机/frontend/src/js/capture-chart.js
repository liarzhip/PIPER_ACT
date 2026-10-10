// Display history is sampled from console status, independently of HDF5 recording.
const names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', '夹爪'];
const colors = ['#2563eb','#e07813','#16a085','#dc405a','#8b5cf6','#64748b','#b38b00'];
const windowSeconds = 60;
export function createJointChart() {
let points = [], origin = null, dataset = null, previousState = null;
let root;

function initCaptureCharts(container) {
  root = container;
  root.innerHTML = `<div class="joint-color-legend">${names.map((name,i)=>`<span style="--joint-color:${colors[i]}"><i></i>${name}</span>`).join('')}</div><canvas role="img" aria-label="J1 至 J6 与夹爪的时间变化曲线；左轴关节 rad，右轴夹爪 m"></canvas>`;
  new ResizeObserver(drawCaptureCharts).observe(root);
  drawCaptureCharts();
}

function updateCaptureCharts(status, now = performance.now() / 1000) {
  if (dataset !== status.dataset_dir || (status.state === 'connecting' && previousState !== 'connecting') ||
      (['ready','recording'].includes(status.state) && ['disconnected','error'].includes(previousState))) {
    points = []; origin = null;
  }
  dataset = status.dataset_dir; previousState = status.state;
  if (status.worker_active && ['ready','recording','saving'].includes(status.state) && status.qpos_hz > 0) {
    if (origin === null) origin = now;
    points.push({t: now - origin, q: [...status.qpos], a: [...status.action]});
    const cutoff = now - origin - windowSeconds;
    while (points.length && points[0].t < cutoff) points.shift();
    // Bound memory even if status polling is made faster in future.
    if (points.length > 600) points.splice(0, points.length - 600);
  }
  drawCaptureCharts();
}

function drawCaptureCharts() {
  if (!root) return;
  const canvas = root.querySelector('canvas'), w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
  const last = points.at(-1), end = Math.max(10,last?.t ?? 0), start = Math.max(0,end-windowSeconds);
  const ranges = [[],[]];
  for (const point of points) for (let i=0;i<7;i++) for (const key of ['q','a']) {
    const v=point[key][i]; if(Number.isFinite(v)) ranges[i===6?1:0].push(v);
  }
  const limits = ranges.map((values,i)=>{
    let low=values.length?Math.min(...values):0, high=values.length?Math.max(...values):(i?.1:1);
    const pad=Math.max((high-low)*.12,i?.001:.01);return [low-pad,high+pad];
  });
  const left=52,right=w-54,top=28,bottom=h-30;
  const x=t=>left+(t-start)/(end-start)*(right-left);
  const y=(v,i)=>{const [low,high]=limits[i===6?1:0];return bottom-(v-low)/(high-low)*(bottom-top);};
  ctx.font='11px Segoe UI';ctx.lineWidth=1;
  ctx.fillStyle='#73838e';ctx.textAlign='left';ctx.fillText('关节 / rad',left,13);
  ctx.textAlign='right';ctx.fillStyle=colors[6];ctx.fillText('夹爪 / m',right,13);
  for(let i=0;i<=4;i++){
    const yy=top+(bottom-top)*i/4;
    ctx.strokeStyle='#e4ebef';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(right,yy);ctx.stroke();
    limits.forEach(([low,high],axis)=>{
      ctx.fillStyle=axis?colors[6]:'#73838e';ctx.textAlign=axis?'left':'right';
      ctx.fillText((high-(high-low)*i/4).toFixed(axis?3:2),axis?right+6:left-6,yy+4);
    });
    const t=start+(end-start)*i/4;ctx.fillStyle='#73838e';ctx.textAlign='center';ctx.fillText(`${t.toFixed(1)} s`,x(t),h-9);
  }
  ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();
  for(let i=0;i<7;i++) for(const key of ['a','q']){
    ctx.strokeStyle=colors[i];ctx.lineWidth=key==='q'?1.8:1.3;ctx.globalAlpha=key==='q'?1:.65;
    ctx.setLineDash(key==='q'?[]:[5,3]);ctx.beginPath();let prior=null;
    for(const point of points){
      const value=point[key][i];if(!Number.isFinite(value)){prior=null;continue;}
      if(prior && point.t-prior.t<2)ctx.lineTo(x(point.t),y(value,i));else ctx.moveTo(x(point.t),y(value,i));
      prior=point;
    }
    ctx.stroke();
    const v=last?.[key][i];if(Number.isFinite(v)){ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(x(last.t),y(v,i),2,0,Math.PI*2);ctx.fill();}
  }
  ctx.restore();ctx.setLineDash([]);ctx.globalAlpha=1;
  if(!points.length){ctx.fillStyle='#8999a4';ctx.textAlign='center';ctx.fillText('等待实时关节数据',(left+right)/2,(top+bottom)/2);}
}

return {init:initCaptureCharts, update:updateCaptureCharts, draw:drawCaptureCharts};
}
const captureChart = createJointChart();
export const initCaptureCharts = captureChart.init;
export const updateCaptureCharts = captureChart.update;
export const drawCaptureCharts = captureChart.draw;
