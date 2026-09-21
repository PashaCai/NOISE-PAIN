/* Sound, Felt. — original PCM audio → channel-power FFT → scalar-field contours.
   No CDN, no server-side processing, no synthesized replacement audio. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const profiles = window.SOUND_PROFILES;
  profiles.forEach(p => { p.title = p.title.toUpperCase(); });
  const audio = $('audio'), canvas = $('terrain'), stage = $('stage');
  const clamp = (n,a,b) => Math.max(a,Math.min(b,n));
  const fmt = n => `${String(Math.floor(n/60)).padStart(2,'0')}:${String(Math.floor(n%60)).padStart(2,'0')}`;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let selected = 0, track = profiles[0], playRequest = 0, uiTick = 0;
  let context, analyser, analyserData, contextSource;
  let zoom = 1, angle = 0, tilt = 0, drag = null, pointer = [0,0], pointerStrength = 0;
  let gl, program, uniforms, fallback;
  let smoothed = new Float32Array(31), lastTime = 0, lastFrameTime = -1;
  let currentFeatures = new Float32Array(31), smoothInitialized = false;
  let errorTimer, waiting = false, lastUIPosition = -1, scrubbing = false;
  audio.volume = .35;

  const vertex = `#version 300 es
  in vec2 position;
  void main(){gl_Position=vec4(position,0.,1.);}`;
  const fragment = `#version 300 es
  precision highp float;
  out vec4 outColor;
  uniform vec2 resolution;
  uniform float time, zoom, angle, tilt, low, mid, high, energy, attack, spectralCenter, flatness, pointerStrength;
  uniform vec2 pointer;
  uniform float spectrum[24];
  uniform float signature[24];
  float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453123);}
  float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1)),f.x),f.y)*2.-1.;}
  float fbm(vec2 p){float v=0.,a=.5;mat2 m=mat2(.8,-.6,.6,.8);for(int i=0;i<4;i++){v+=a*noise(p);p=m*p*2.03+4.71;a*=.5;}return v;}
  mat2 rot(float a){return mat2(cos(a),-sin(a),sin(a),cos(a));}
  void main(){
    vec2 uv=(gl_FragCoord.xy-.5*resolution)/min(resolution.x,resolution.y)*2.;
    vec2 p=uv/(zoom*.70);
    p.y+=.012;
    p=rot(angle)*p;
    p.x*=1.+tilt*.21; p.y*=1.-tilt*.13;
    vec2 delta=p-pointer;
    float influence=exp(-dot(delta,delta)*3.2)*pointerStrength;
    p+=delta*influence*.24;
    float brightness=clamp(log(1.+spectralCenter*20000.)/10.,.0,1.);
    float phase=signature[2]*13.+signature[10]*7.+signature[19]*17.;
    float speed=time*.24;
    vec2 q=p;
    // Low energy deforms the whole mass. High energy bends it at finer scales.
    q.x+=.09*low*sin(p.y*4.+speed*1.2+phase)*energy;
    q.y+=.075*low*cos(p.x*3.-speed+phase)*energy;
    q=rot(.1*sin(phase))*q;
    q.x*=.90+brightness*.20;
    float r=length(q), a=atan(q.y,q.x);
    float petals=0.;
    for(int i=0;i<12;i++){
      float k=float(i)+2.;
      float amp=signature[i*2]*.7+spectrum[i*2]*.3;
      petals+=amp*sin(a*k+phase*k*.19+speed*(.2+float(i)*.034)+r*float(i)*.17)/sqrt(k);
    }
    float coarse=fbm(q*(2.0+mid*2.)+vec2(phase,speed*.21));
    float folds=fbm(q*(5.+mid*5.)+coarse*.8+vec2(-speed*.17,phase));
    float fine=fbm(q*(18.+high*18.)+vec2(phase,-speed*.4));
    float pulse=attack*.16*sin(r*12.-time*2.);
    float interior=1.-smoothstep(.35,1.12,r);
    float field=.43+energy*.10-r + petals*(.17+energy*.075)+coarse*(.16+low*.12)+folds*(.025+mid*.075)*interior
      +fine*high*(.012+energy*.026)*(.4+.6*interior)*(1.+flatness*.25)+pulse;
    field+=.025*energy*sin(r*7.-speed*2.3)*low;
    float density=37.+high*22.+mid*6.;
    float v=field*density;
    float d=abs(fract(v+.5)-.5);
    float aa=max(fwidth(v),.012);
    float line=1.-smoothstep(.042,.042+aa*.92,d);
    float edge=smoothstep(-.61,-.43,field);
    float opacity=mix(.18,.86,smoothstep(-.52,.20,field));
    float ink=line*opacity*edge;
    // Discrete pigment in the core echoes the original openFrameworks work.
    vec2 cell=floor(q*(190.+high*70.));
    float grain=hash(cell+floor(phase*20.));
    float holes=fbm(cell*.08+vec2(speed*.34,phase));
    float core=smoothstep(.045,.26,field);
    float pigment=step(.25+holes*.38+high*.08,grain);
    ink=max(ink,core*pigment*(.87+.11*energy));
    float fineRing=1.-smoothstep(.035,.035+aa,abs(fract(v*1.8+.5)-.5));
    ink=max(ink,fineRing*core*.61);
    // Tiny detached marks remain part of the same field rather than random confetti.
    float dust=step(.997,grain)*smoothstep(-.55,-.12,field)*(1.-smoothstep(-.10,.13,field));
    ink=max(ink,dust*.2);
    vec3 paper=vec3(.972549,.968627,.952941);
    vec3 blue=vec3(.12549,.172549,1.);
    outColor=vec4(mix(paper,blue,clamp(ink,0.,1.)),1.);
  }`;

  function initGraphics(){
    gl=canvas.getContext('webgl2',{antialias:false,alpha:false,powerPreference:'high-performance',preserveDrawingBuffer:false});
    if(!gl){initFallback();return;}
    function compile(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
    try{
      program=gl.createProgram();gl.attachShader(program,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(program,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);
      if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
      gl.useProgram(program);
      const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
      const loc=gl.getAttribLocation(program,'position');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
      uniforms={};for(const name of ['resolution','time','zoom','angle','tilt','low','mid','high','energy','attack','spectralCenter','flatness','pointer','pointerStrength','spectrum','signature'])uniforms[name]=gl.getUniformLocation(program,name);
      canvas.dataset.renderer='webgl2';
    }catch(error){console.error(error);gl=null;initFallback();}
  }
  function initFallback(){
    // A 2D contour renderer keeps audio and interaction available without WebGL.
    const replacement=canvas.cloneNode();canvas.replaceWith(replacement);
    fallback={canvas:replacement,ctx:replacement.getContext('2d')};
    replacement.dataset.renderer='canvas2d';
  }
  function surface(){return fallback?fallback.canvas:canvas;}
  function resize(){
    const c=surface(), box=stage.getBoundingClientRect();
    const ratio=Math.min(devicePixelRatio||1,1.6);
    c.width=Math.round(box.width*ratio);c.height=Math.round(box.height*ratio);
    if(gl)gl.viewport(0,0,c.width,c.height);
    drawWave();
  }
  function renderFallback(t,f){
    const {canvas:c,ctx}=fallback,w=c.width,h=c.height;
    ctx.fillStyle='#f8f7f3';ctx.fillRect(0,0,w,h);ctx.save();ctx.translate(w/2,h/2);ctx.rotate(angle);ctx.scale(zoom*(1+tilt*.1),zoom*(1-tilt*.1));
    const base=Math.min(w,h)*.25;
    for(let ring=0;ring<48;ring++){
      const rr=base*(.11+ring/45), s=track.signature;
      ctx.beginPath();
      for(let i=0;i<=300;i++){
        const a=i/300*Math.PI*2;let d=0;
        for(let k=0;k<12;k++)d+=s[k*2]*Math.sin(a*(k+2)+s[4]*10+k*.3+t*.1)/Math.sqrt(k+2);
        let r=rr*(1+.22*d)+Math.sin(a*5+t*.2)*base*f[1]*.08+Math.sin(a*29+t*.8)*base*f[3]*.014;
        r+=base*f[6]*.10;const x=Math.cos(a)*r,y=Math.sin(a)*r;
        i?ctx.lineTo(x,y):ctx.moveTo(x,y);
      }
      ctx.closePath();ctx.strokeStyle=`rgba(32,44,255,${.88-ring*.014})`;ctx.lineWidth=Math.max(.7,w/1100);ctx.stroke();
    }
    ctx.restore();
  }

  function sampleFeatures(position,preview=false){
    const frames=track.frames;
    if(preview){
      // Show the actual track fingerprint before playback, including recordings
      // whose first samples are silent. Once playing, only the current time is used.
      currentFeatures.fill(0);currentFeatures[0]=.65;
      currentFeatures.set(track.proportions,1);currentFeatures[4]=track.centroid/20000;currentFeatures.set(track.signature,7);
      return currentFeatures;
    }
    const index=clamp(position*track.fps,0,frames.length-1),a=Math.floor(index),b=Math.min(a+1,frames.length-1),k=index-a;
    for(let i=0;i<31;i++)currentFeatures[i]=frames[a][i]*(1-k)+frames[b][i]*k;
    return currentFeatures;
  }

  function drawFrame(now){
    requestAnimationFrame(drawFrame);
    if(document.hidden)return;
    if(now-lastTime<1000/(reduced?24:40))return;
    const dt=Math.min((now-lastTime)/1000,.12);lastTime=now;
    const playing=!audio.paused&&!audio.ended;
    const t=Number.isFinite(audio.currentTime)?audio.currentTime:0;
    const f=sampleFeatures(t,t===0&&!playing);
    const blend=1-Math.exp(-dt*11);
    for(let i=0;i<31;i++)smoothed[i]=smoothInitialized?smoothed[i]+(f[i]-smoothed[i])*blend:f[i];
    smoothInitialized=true;
    const s=smoothed;
    pointerStrength+=( (drag?1:0)-pointerStrength)*Math.min(dt*5,1);
    if(gl){
      gl.useProgram(program);
      gl.uniform2f(uniforms.resolution,canvas.width,canvas.height);
      gl.uniform1f(uniforms.time,reduced?t*.25:t);
      gl.uniform1f(uniforms.zoom,zoom);gl.uniform1f(uniforms.angle,angle);gl.uniform1f(uniforms.tilt,tilt);
      gl.uniform1f(uniforms.low,s[1]);gl.uniform1f(uniforms.mid,s[2]);gl.uniform1f(uniforms.high,s[3]);
      gl.uniform1f(uniforms.energy,s[0]*(reduced?.5:1));gl.uniform1f(uniforms.attack,clamp(s[6]*3,0,1)*(reduced?.2:1));
      gl.uniform1f(uniforms.spectralCenter,track.centroid/20000);gl.uniform1f(uniforms.flatness,s[5]);
      gl.uniform2f(uniforms.pointer,pointer[0],pointer[1]);gl.uniform1f(uniforms.pointerStrength,pointerStrength);
      gl.uniform1fv(uniforms.spectrum,s.subarray(7));gl.uniform1fv(uniforms.signature,new Float32Array(track.signature));
      gl.drawArrays(gl.TRIANGLES,0,6);
    }else if(fallback)renderFallback(t,s);
    if(now-uiTick>100){updateUI(s,t);uiTick=now;}
    lastFrameTime=t;
  }

  function updateUI(f,t){
    const silent=f[0]<.005;
    ['low','mid','high'].forEach((key,i)=>{const value=silent?0:clamp(f[i+1]*100,0,100);$(key+'Value').textContent=`${Math.round(value)}%`;$(key+'Meter').style.width=`${value}%`;});
    $('currentTime').textContent=fmt(t);
    if(!scrubbing)$('seek').value=t;
    $('seek').setAttribute('aria-valuetext',`${fmt(t)} / ${fmt(track.duration)}`);
    const low=f[1],high=f[3];
    $('textureDescription').textContent=silent?'A moment of quiet':high>.55?'Fine, tense edges':low>.55?'Heavy, sustained pressure':'Layers of movement';
    if(Math.abs(t-lastUIPosition)>.06||t===0){drawWave();lastUIPosition=t;}
  }
  function drawWave(){
    const c=$('waveform'),rect=c.getBoundingClientRect();if(!rect.width)return;
    const ratio=Math.min(devicePixelRatio||1,2);const w=Math.round(rect.width*ratio),h=Math.round(rect.height*ratio);
    if(c.width!==w||c.height!==h){c.width=w;c.height=h;}
    const ctx=c.getContext('2d');ctx.clearRect(0,0,w,h);
    const count=Math.min(180,Math.floor(rect.width/3)),frac=audio.currentTime/track.duration;
    for(let i=0;i<count;i++){
      const index=Math.floor(i/count*track.waveform.length),height=(.12+.88*Math.pow(track.waveform[index],.6))*h;
      ctx.fillStyle=i/count<=frac?'#202cff':'#cdd0c2';
      ctx.fillRect(i/count*w,(h-height)/2,Math.max(ratio,w/count-ratio*1.4),height);
    }
  }

  function setState(text,playing=false){$('stateText').textContent=text;$('stateDot').classList.toggle('playing',playing);}
  function syncPlayback(){
    const playing=!audio.paused&&!audio.ended;
    $('playIcon').innerHTML=playing?'<path d="M6 5H10V19H6ZM14 5H18V19H14Z"/>':'<path d="M8 5L19 12L8 19Z"/>';
    $('playButton').setAttribute('aria-label',playing?'Pause audio':'Play audio');
    if(playing)setState(waiting?'Buffering':audio.muted?'Watching without sound':'Sound in motion',true);
    else setState(audio.ended?'Recording ended':'Paused');
    document.querySelectorAll('.sound-item').forEach((el,i)=>{el.querySelector('.sound-symbol').textContent=i===selected?(playing?'Ⅱ':'↗'):'';});
  }
  async function enableAnalysis(){
    // The offline FFT drives the full visual. A Web Audio tap is available while
    // listening over HTTP and never depends on the user's volume or microphone.
    if(location.protocol==='file:')return;
    try{
      if(!context){
        context=new (window.AudioContext||window.webkitAudioContext)();
        contextSource=context.createMediaElementSource(audio);
        analyser=context.createAnalyser();analyser.fftSize=2048;analyser.smoothingTimeConstant=.8;
        analyserData=new Uint8Array(analyser.frequencyBinCount);
        contextSource.connect(analyser);analyser.connect(context.destination);
      }
      if(context.state==='suspended')await context.resume();
    }catch(e){console.warn('Optional Web Audio analyser unavailable',e);}
  }
  async function startPlayback(userGesture=false){
    const request=++playRequest;
    if(userGesture){enableAnalysis();$('entry').hidden=true;}
    clearError();
    try{
      if(context&&context.state==='suspended'&&userGesture)await context.resume();
      await audio.play();
      if(request!==playRequest)return;
      $('entry').hidden=true;syncPlayback();
    }catch(e){
      if(request!==playRequest||e.name==='AbortError')return;
      if(e.name==='NotAllowedError'){$('entry').hidden=false;setState('Click to enable audio');}
      else showError('This recording could not be played. Please try again.');
    }
  }
  function clearError(){clearTimeout(errorTimer);$('errorMessage').hidden=true;}
  function showError(message){
    $('entry').hidden=true;const el=$('errorMessage');el.replaceChildren(document.createTextNode(message));
    const b=document.createElement('button');b.textContent='Reload';b.onclick=()=>{audio.load();startPlayback(true);};el.append(b);el.hidden=false;setState('Audio failed to load');
  }
  function selectTrack(index,auto=true,gesture=false){
    playRequest++;audio.pause();selected=index;track=profiles[index];smoothInitialized=false;waiting=false;lastUIPosition=-1;
    clearError();audio.src=track.file;audio.load();
    $('trackTitle').textContent=track.title;$('trackFile').textContent=`${track.id}.wav`;
    $('playerLabel').textContent=`${String(track.id).padStart(2,'0')} / ${track.title}`;
    $('artNumber').textContent=`FIG. ${String(track.id).padStart(2,'0')}`;
    $('shapeLabel').textContent=track.dominant==='Low-dominant'?'Low / Pressure and swells':track.dominant==='High-dominant'?'High / Grain and sharpness':'Mid / Folds and overlap';
    $('duration').textContent=fmt(track.duration);$('currentTime').textContent='00:00';$('seek').max=track.duration;$('seek').value=0;
    document.querySelectorAll('.sound-item').forEach((el,i)=>{el.classList.toggle('active',i===index);el.setAttribute('aria-pressed',i===index?'true':'false');});
    setState('Loading recording');drawWave();
    if(auto)startPlayback(gesture);
  }
  function togglePlayback(){
    if(audio.paused||audio.ended){if(audio.ended)audio.currentTime=0;startPlayback(true);}
    else{playRequest++;audio.pause();}
  }

  profiles.forEach((p,i)=>{
    const b=document.createElement('button');b.className='sound-item';b.type='button';b.setAttribute('aria-pressed','false');b.setAttribute('aria-label',`${p.title}, ${p.dominant}, ${fmt(p.duration)}, Select and play`);
    b.innerHTML=`<span class="sound-index">${String(p.id).padStart(2,'0')}</span><span class="sound-name">${p.title}<small>${p.dominant.replace('-dominant','')}</small></span><span class="sound-duration">${fmt(p.duration)}</span><span class="sound-symbol" aria-hidden="true"></span>`;
    b.onclick=()=>selectTrack(i,true,true);
    b.onkeydown=e=>{if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();const n=(i+(e.key==='ArrowDown'?1:-1)+profiles.length)%profiles.length;$('soundList').children[n].focus();selectTrack(n,true,true);}};
    $('soundList').append(b);
  });
  $('playButton').onclick=togglePlayback;
  $('enterButton').onclick=()=>startPlayback(true);
  $('seek').oninput=e=>{const t=Number(e.target.value);if(Number.isFinite(audio.duration))audio.currentTime=clamp(t,0,audio.duration);smoothInitialized=false;drawWave();$('currentTime').textContent=fmt(t);};
  $('seek').addEventListener('pointerdown',()=>scrubbing=true);
  window.addEventListener('pointerup',()=>scrubbing=false);
  window.addEventListener('pointercancel',()=>scrubbing=false);
  $('seek').onchange=()=>{scrubbing=false;};
  $('seek').addEventListener('keydown',e=>{
    if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
      e.preventDefault();audio.currentTime=clamp(audio.currentTime+(e.key==='ArrowRight'?1:-1),0,track.duration);$('seek').value=audio.currentTime;smoothInitialized=false;
    }
  });
  function syncVolume(){
    $('muteButton').setAttribute('aria-pressed',String(audio.muted));$('muteButton').setAttribute('aria-label',audio.muted?'Unmute':'Mute');
    $('volumeValue').textContent=audio.muted?'0%':`${Math.round(audio.volume*100)}%`;document.body.classList.toggle('is-muted',audio.muted);syncPlayback();
  }
  $('volume').oninput=e=>{audio.volume=Number(e.target.value)/100;audio.muted=audio.volume===0;syncVolume();};
  $('muteButton').onclick=()=>{audio.muted=!audio.muted;syncVolume();};
  $('loopButton').onclick=()=>{audio.loop=!audio.loop;$('loopButton').setAttribute('aria-pressed',String(audio.loop));};
  audio.addEventListener('play',()=>{waiting=false;syncPlayback();});audio.addEventListener('pause',syncPlayback);audio.addEventListener('ended',syncPlayback);
  audio.addEventListener('playing',()=>{waiting=false;clearTimeout(errorTimer);syncPlayback();});
  audio.addEventListener('waiting',()=>{waiting=true;if(!audio.paused)setState('Buffering',true);});
  audio.addEventListener('error',()=>{if(audio.error)showError('Unable to load audio. Check that the audio folder contains all recordings.');});
  audio.addEventListener('loadedmetadata',()=>{if(Math.abs(audio.duration-track.duration)>.5)console.warn('Duration mismatch');});

  function updateZoom(value){zoom=clamp(value,.55,1.8);$('zoomValue').textContent=`${Math.round(zoom*100)}%`;}
  function resetView(){angle=0;tilt=0;pointer=[0,0];pointerStrength=0;updateZoom(1);}
  $('zoomIn').onclick=()=>updateZoom(zoom+.1);$('zoomOut').onclick=()=>updateZoom(zoom-.1);$('resetView').onclick=resetView;
  $('aboutButton').onclick=()=>{$('aboutDialog').showModal();};$('closeAbout').onclick=()=>$('aboutDialog').close();
  $('aboutDialog').onclick=e=>{if(e.target===$('aboutDialog')){const r=$('aboutDialog').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('aboutDialog').close();}};
  document.addEventListener('keydown',e=>{
    if(e.code==='Escape'){playRequest++;audio.pause();return;}
    if($('aboutDialog').open)return;
    if(e.code==='Space'&&!['INPUT','BUTTON','A'].includes(document.activeElement.tagName)){e.preventDefault();togglePlayback();}
  });
  initGraphics();
  const c=surface();
  c.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={id:e.pointerId,x:e.clientX,y:e.clientY};c.setPointerCapture(e.pointerId);});
  c.addEventListener('pointermove',e=>{
    if(!drag||e.pointerId!==drag.id)return;
    const rect=c.getBoundingClientRect();angle+=(e.clientX-drag.x)*.005;tilt=clamp(tilt+(e.clientY-drag.y)*.006,-1,1);drag.x=e.clientX;drag.y=e.clientY;
    const x=(e.clientX-rect.left-rect.width/2)/Math.min(rect.width,rect.height)*2/(zoom*.7),y=-(e.clientY-rect.top-rect.height/2)/Math.min(rect.width,rect.height)*2/(zoom*.7);
    pointer=[Math.cos(angle)*x+Math.sin(angle)*y,-Math.sin(angle)*x+Math.cos(angle)*y];
  });
  const release=e=>{if(drag&&e.pointerId===drag.id)drag=null;};c.addEventListener('pointerup',release);c.addEventListener('pointercancel',release);c.addEventListener('lostpointercapture',()=>drag=null);
  c.addEventListener('wheel',e=>{e.preventDefault();updateZoom(zoom*Math.exp(-e.deltaY*.001));},{passive:false});
  c.addEventListener('dblclick',resetView);
  c.addEventListener('keydown',e=>{if(e.key==='+'||e.key==='='){e.preventDefault();updateZoom(zoom+.1);}if(e.key==='-'){e.preventDefault();updateZoom(zoom-.1);}if(e.key==='0'){e.preventDefault();resetView();}});
  c.addEventListener('webglcontextlost',e=>{e.preventDefault();setState('Graphics interrupted. Please refresh the page.');});
  c.addEventListener('webglcontextrestored',()=>location.reload());
  new ResizeObserver(resize).observe(stage);
  window.addEventListener('resize',drawWave);
  selectTrack(0,true,false);resize();requestAnimationFrame(drawFrame);
})();

