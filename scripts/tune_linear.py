"""Benchmark FP32 Vulkan GEMM tiles using D3's dominant actual matrix sizes."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from vulkan_linear import VulkanEngine

ROOT=Path(__file__).resolve().parents[1]


def shader(bm,bn,micro_m,bk,pad):
    tx=bn//4;ty=bm//micro_m;group=tx*ty
    return f'''#version 450
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x={tx},local_size_y={ty}) in;
layout(set=0,binding=0,std430) readonly buffer Input {{ float x[]; }};
layout(set=0,binding=1,std430) readonly buffer Weight {{ float w[]; }};
layout(set=0,binding=2,std430) writeonly buffer Output {{ float y[]; }};
layout(push_constant) uniform Dims {{ uint M;uint N;uint K; }} d;
shared float aa[{bm*(bk+pad)}];
shared float bb[{bk*bn}];
void main() {{
 uint tx=gl_LocalInvocationID.x,ty=gl_LocalInvocationID.y;
 uint tid=ty*{tx}+tx,bm=gl_WorkGroupID.y*{bm},bn=gl_WorkGroupID.x*{bn};
 vec4 acc[{micro_m}];
 [[unroll]] for(uint i=0;i<{micro_m};i++)acc[i]=vec4(0);
 for(uint base=0;base<d.K;base+={bk}) {{
  [[unroll]] for(uint z=tid;z<{bm*bk};z+={group}) {{
   uint row=z/{bk},col=z%{bk};
   aa[row*{bk+pad}+col]=(bm+row<d.M && base+col<d.K)?x[(bm+row)*d.K+base+col]:0.;
  }}
  [[unroll]] for(uint z=tid;z<{bn*bk};z+={group}) {{
   uint row=z/{bk},col=z%{bk};
   bb[col*{bn}+row]=(bn+row<d.N && base+col<d.K)?w[(bn+row)*d.K+base+col]:0.;
  }}
  barrier();
  [[unroll]] for(uint k=0;k<{bk};k++) {{
   vec4 bv=vec4(bb[k*{bn}+tx],bb[k*{bn}+tx+{tx}],bb[k*{bn}+tx+{2*tx}],bb[k*{bn}+tx+{3*tx}]);
   [[unroll]] for(uint i=0;i<{micro_m};i++)acc[i]=fma(vec4(aa[(ty+i*{ty})*{bk+pad}+k]),bv,acc[i]);
  }}
  barrier();
 }}
 [[unroll]] for(uint i=0;i<{micro_m};i++) {{
  uint row=bm+ty+i*{ty};
  [[unroll]] for(uint j=0;j<4;j++) {{
   uint col=bn+tx+j*{tx};
   if(row<d.M && col<d.N)y[row*d.N+col]=acc[i][j]+w[d.N*d.K+col];
  }}
 }}
}}
'''


def main():
    torch.set_num_threads(4)
    folder=ROOT/'results/cache/gemm_tuning';folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(812)
    shapes=[(619,3584,1024),(619,1024,3584),(619,6144,1024)]
    weights=[24,12,6]
    inputs=[]
    for m,n,k in shapes:
        x=torch.from_numpy((rng.normal(size=(m,k))*.04).astype('float32'))
        linear=torch.nn.Linear(k,n,bias=True)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy((rng.normal(size=(n,k))*.04).astype('float32')))
            linear.bias.copy_(torch.from_numpy((rng.normal(size=n)*.01).astype('float32')))
            reference=linear(x).numpy()
        inputs.append((x,linear,reference))
    configs=[(64,64,4,16,0),(32,32,4,16,0),(32,64,4,16,0),(64,32,4,16,0),
             (64,64,8,16,0),(64,64,4,32,0),(64,64,4,8,0),(64,64,4,16,1),
             (64,64,4,32,1),(64,64,8,16,1),(64,64,8,32,1),(32,64,4,32,1),
             (32,32,4,32,1),(64,128,8,16,1),(128,64,8,16,1),(16,64,4,16,1)]
    reports=[]
    for config in configs:
        bm,bn,micro_m,bk,pad=config
        name='_'.join(map(str,config));source=folder/f'{name}.comp';binary=source.with_suffix('.spv')
        source.write_text(shader(*config))
        subprocess.run(['glslc','-O',str(source),'-o',str(binary)],check=True)
        library=ROOT/'native/liblinear_tuned.so'
        if not library.exists():library=ROOT/'native/liblinear.so'
        engine=VulkanEngine(shader_path=binary,tile_m=bm,tile_n=bn,library_path=library)
        durations=[];errors=[]
        try:
            for x,layer,reference in inputs:
                index=engine.add(layer)
                out=engine.run(index,x,layer.out_features).numpy()
                np.testing.assert_allclose(out,reference,atol=2e-5,rtol=2e-4)
                errors.append(float(np.max(abs(out-reference))))
                times=[]
                for _ in range(3):
                    start=time.perf_counter();engine.run(index,x,layer.out_features);times.append(time.perf_counter()-start)
                durations.append(float(np.median(times)))
        finally:engine.close()
        row={'config':config,'shader':str(source),'shapes':shapes,'median_seconds':durations,
             'weighted_main_projection_seconds':sum(a*b for a,b in zip(durations,weights)),
             'max_absolute_errors':errors}
        reports.append(row);print(json.dumps(row),flush=True)
        (ROOT/'results/linear_tile_tuning.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
