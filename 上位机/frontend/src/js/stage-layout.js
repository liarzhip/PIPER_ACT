// Relocate existing controls, preserving their IDs, form ownership and handlers.
export function setupStageLayouts() {
  for (const [phase,formId,ids,label] of [
    ['capture','configForm',['connect','disconnect','start','stop'],'采集参数与保存目录'],
    ['training','trainingForm',['startTraining','stopTraining'],'训练参数与模型目录'],
    ['execution','executionForm',['startExecution','stopExecution'],'执行参数与模型目录']
  ]) {
    const page=document.getElementById(`page-${phase}`), form=document.getElementById(formId);
    page.classList.add('operation-page');
    const toolbar=document.createElement('div'); toolbar.className='operation-toolbar'; toolbar.setAttribute('aria-label',`${label.slice(0,2)}操作`);
    const actions=document.createElement('div'); actions.className='operation-buttons';
    for(const id of ids){const button=document.getElementById(id);if(['connect','startTraining','startExecution'].includes(id)){button.type='submit';button.setAttribute('form',formId);}else{button.type='button';button.removeAttribute('form');}actions.append(button);}
    toolbar.append(actions);
    const settings=document.createElement('details');settings.className='stage-settings';
    const summary=document.createElement('summary');summary.textContent=label+'（点击展开）';settings.append(summary);
    form.before(settings);settings.append(form);
    const toggle=document.createElement('button');toggle.className='button ghost';toggle.type='button';toggle.textContent='参数设置';
    toggle.onclick=()=>{settings.open=!settings.open;if(settings.open)settings.scrollIntoView({block:'start',behavior:'smooth'});};toolbar.append(toggle);
    form.addEventListener('invalid',event=>{
      for(let node=event.target.parentElement;node;node=node.parentElement)if(node.tagName==='DETAILS')node.open=true;
    },true);
    page.querySelector('.stage-banner').after(toolbar);
    const log=page.querySelector('.log-card');
    if(log){const details=document.createElement('details');details.className='stage-log';const title=document.createElement('summary');title.textContent='运行日志（点击展开）';log.before(details);details.append(title,log);}
    if(phase==='capture'){
      const title=document.getElementById('recordingTitle');title.classList.add('operation-status');toolbar.prepend(title);
      page.querySelector('.camera-card h2').textContent='双相机实时画面';
      const note= form.querySelector('.path-parameters .parameter-note');if(note)note.textContent='可修改数据集保存目录，连接时应用；已有文件不会搬动。';
    }
    if(phase==='execution'){
      const mode=form.querySelector('.mode-picker'), safety=document.getElementById('safetyBox');
      for(const element of [...mode.querySelectorAll('input'),...safety.querySelectorAll('input')])element.setAttribute('form',formId);
      const options=document.createElement('div');options.className='execution-quick-mode';options.append(mode,safety);toolbar.prepend(options);
    }
  }
}
