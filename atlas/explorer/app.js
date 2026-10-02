'use strict';
// Reserve room for the hosting site's floating controls when embedded.
if(window.self!==window.top)document.body.classList.add('hf-embedded');
const REPO='https://huggingface.co/datasets/lamm-mit/graphene-design-universe-256k/resolve/v1.0.0/';
const $=id=>document.getElementById(id);
let designs=[],groups=[],colors=[],selected=-1,generation=0,filtered=[],loaded=false;
const imageCache=new Map();
// Compact row schema: id, group, atoms, Lx_nm, Ly_nm, porosity, x, y, z,
// coordinate_offset, coordinate_size, image_offset, image_size, coordinate_SHA256.
async function gunzip(bytes){const signature=new Uint8Array(bytes);if(signature[0]!==31||signature[1]!==139)return bytes;return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();}
async function member(path,offset,size,key){
 const response=await fetch(REPO+path+'?member='+key,{headers:{Range:`bytes=${offset}-${offset+size-1}`}});
 if(!response.ok)throw Error(`Download failed (${response.status})`);
 const data=await response.arrayBuffer();
 if(response.status===206){if(data.byteLength!==size)throw Error('Incomplete file range');return data;}
 if(data.byteLength<offset+size)throw Error('Incomplete file');return data.slice(offset,offset+size);
}
function shard(id){return String(Math.floor(id/256)).padStart(4,'0');}
async function imageURL(id){
 if(imageCache.has(id))return imageCache.get(id);
 const r=designs[id];const task=member(`images/previews-${shard(id)}.tar`,r[11],r[12],id).then(data=>URL.createObjectURL(new Blob([data],{type:'image/png'})));
 imageCache.set(id,task);
 try{return await task;}catch(e){imageCache.delete(id);throw e;}
}
function makeTrace(ids,group){return {type:'scatter3d',mode:'markers',name:groups[group],x:ids.map(i=>designs[i][6]),y:ids.map(i=>designs[i][7]),z:ids.map(i=>designs[i][8]),customdata:ids,text:ids.map(i=>`GDU-${String(i).padStart(6,'0')}<br>${groups[group]}<br>${designs[i][2].toLocaleString()} C atoms`),hovertemplate:'%{text}<extra></extra>',marker:{size:2.3,color:colors[group],opacity:.85},showlegend:false};}
function highlight(){if(selected<0||!filtered.includes(selected))return {type:'scatter3d',mode:'markers',x:[],y:[],z:[],showlegend:false};const r=designs[selected];return {type:'scatter3d',mode:'markers',x:[r[6]],y:[r[7]],z:[r[8]],customdata:[selected],marker:{size:6,color:'#ffffff',opacity:1},showlegend:false,hovertemplate:`GDU-${String(selected).padStart(6,'0')}<extra></extra>`};}
const camera={eye:{x:1.4,y:1.2,z:.9},up:{x:0,y:0,z:1}};
const layout={paper_bgcolor:'#05080d',plot_bgcolor:'#05080d',font:{family:'Arial',color:'#c9d8e6'},margin:{l:0,r:0,t:0,b:0},uirevision:'geometry-map',hoverlabel:{bgcolor:'#15202d',font:{size:11}},scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},bgcolor:'#05080d',aspectmode:'data',camera},showlegend:false};
async function draw(){
 const chosen=$('group').value,population=$('population').value,maxwidth=Number($('size').value),maxvoid=Number($('porosity').value)/100;
 $('sizeLabel').textContent=maxwidth+' nm';$('porosityLabel').textContent=Math.round(maxvoid*100)+'%';
 filtered=designs.filter(r=>(chosen==='all'||r[1]===Number(chosen))&&(population==='all'||(population==='original'?r[0]<64000:r[0]>=64000))&&r[3]<=maxwidth&&r[5]<=maxvoid).map(r=>r[0]);
 $('count').textContent=filtered.length.toLocaleString();
 const buckets=groups.map(()=>[]);filtered.forEach(id=>buckets[designs[id][1]].push(id));
 const traces=buckets.map((ids,g)=>makeTrace(ids,g));traces.push(highlight());
 await Plotly.react('map',traces,layout,{responsive:true,displayModeBar:false,scrollZoom:true,plotGlPixelRatio:Math.min(window.devicePixelRatio,1.5)});
 if(loaded&&selected>=0&&filtered.length&&!filtered.includes(selected))await select(filtered[0]);
}
async function select(id){
 id=Number(id);if(!Number.isInteger(id)||id<0||id>=designs.length)return;
 selected=id;const serial=++generation,r=designs[id];$('design').value=id;
 if(loaded&&!filtered.includes(id)){$('group').value='all';$('population').value='all';$('size').value=500;$('porosity').value=100;await draw();}
 $('neighbors').replaceChildren();
 $('design-title').textContent=groups[r[1]];$('design-id').textContent=`GDU-${String(id).padStart(6,'0')} · unrelaxed geometry`;
 $('atoms').textContent=r[2].toLocaleString();$('cell').textContent=`${r[3].toFixed(2)} × ${r[4].toFixed(2)} nm`;$('void').textContent=(r[5]*100).toFixed(1)+'%';$('winding').textContent=r[14]+' / 2';$('cleanup').textContent=r[15]?'Present':'None';
 $('preview-status').textContent='Loading structure…';$('preview-status').hidden=false;$('preview').style.opacity=.2;$('download').disabled=false;$('download-status').textContent='';
 try{const url=await imageURL(id);if(serial!==generation)return;$('preview').src=url;$('preview').alt=`${groups[r[1]]}, GDU-${String(id).padStart(6,'0')}`;$('preview').style.opacity=1;$('preview-status').hidden=true;}catch(e){if(serial===generation)$('preview-status').textContent='Preview unavailable · try again';}
 if(serial!==generation)return;
 const near=[];let largest=Infinity;for(const q of designs){if(q[0]===id)continue;const d=(q[6]-r[6])**2+(q[7]-r[7])**2+(q[8]-r[8])**2;if(d>=largest)continue;near.push({id:q[0],d});near.sort((a,b)=>a.d-b.d||a.id-b.id);if(near.length>9)near.pop();if(near.length===9)largest=near[8].d;}
 $('neighbors').replaceChildren();for(const n of near){const button=document.createElement('button'),im=document.createElement('img'),label=document.createElement('span');button.title=`GDU-${String(n.id).padStart(6,'0')} · ${groups[designs[n.id][1]]}`;button.setAttribute('aria-label',button.title);im.alt=groups[designs[n.id][1]];label.textContent=String(n.id).padStart(6,'0');button.append(im,label);button.onclick=()=>select(n.id);$('neighbors').append(button);imageURL(n.id).then(url=>{im.src=url;}).catch(()=>{label.textContent='retry';});}
 if(loaded)await Plotly.restyle('map',{x:[[r[6]]],y:[[r[7]]],z:[[r[8]]],customdata:[[id]],hovertemplate:`GDU-${String(id).padStart(6,'0')}<extra></extra>`},[groups.length]);
}
async function download(){
 const id=selected,r=designs[id];$('download').disabled=true;$('download-status').textContent=`Fetching ${r[10]<1024*1024?(r[10]/1024).toFixed(1)+' KB':(r[10]/1024/1024).toFixed(2)+' MB'} of compressed coordinates…`;
 try{const compressed=await member(`coordinates/structures-${shard(id)}.tar`,r[9],r[10],id);const raw=await gunzip(compressed);
 const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw))).map(x=>x.toString(16).padStart(2,'0')).join('');if(hash!==r[13])throw Error('Coordinate checksum mismatch');
 const url=URL.createObjectURL(new Blob([raw],{type:'chemical/x-xyz'})),a=document.createElement('a');a.href=url;a.download=`GDU-${String(id).padStart(6,'0')}.extxyz`;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);$('download-status').textContent='Coordinates verified and downloaded';
 }catch(e){$('download-status').textContent=e.message;}finally{$('download').disabled=false;}
}
async function init(){
 try{const res=await fetch('designs.json.gz');if(!res.ok)throw Error('Map data unavailable');const payload=JSON.parse(new TextDecoder().decode(await gunzip(await res.arrayBuffer())));designs=payload.rows;groups=payload.groups;colors=payload.colors;
 groups.forEach((name,i)=>{const o=document.createElement('option');o.value=i;o.textContent=name;$('group').append(o);});
 await draw();loaded=true;document.querySelectorAll('.toolbar select,.toolbar input,.toolbar button,#search input,#search button').forEach(el=>{el.disabled=false;});$('loading').hidden=true;$('map').on('plotly_click',e=>{const id=e.points[0].customdata;if(Number.isInteger(id))select(id);});await select(1226);
 }catch(e){$('loading').textContent=e.message+' · reload to retry';console.error(e);}
}
$('group').onchange=draw;$('population').onchange=draw;$('size').onchange=draw;$('porosity').onchange=draw;
$('size').oninput=()=>{$('sizeLabel').textContent=$('size').value+' nm';};$('porosity').oninput=()=>{$('porosityLabel').textContent=$('porosity').value+'%';};
$('search').onsubmit=e=>{e.preventDefault();select($('design').value);};$('download').onclick=download;
$('reset').onclick=async()=>{$('group').value='all';$('population').value='all';$('size').value=500;$('porosity').value=100;await draw();Plotly.relayout('map',{'scene.camera':camera});};
$('about').onclick=()=>$('about-dialog').showModal();$('close-about').onclick=()=>$('about-dialog').close();
window.addEventListener('resize',()=>{if(loaded)Plotly.Plots.resize('map');});init();
