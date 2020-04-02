.. _sec-gpu:

Appendix C: GPU compiler
========================

.. contents::
   :local:
   :depth: 4

Basicly CPU compiler is SISD (Single Instruction Single Data Architecture). 
The vector or multimedia instructions in CPU are small scaled of SIMD
(Single Instruction Multiple Data) for 4 or 16 element while GPU is a large 
scaled of SIMD processor needing to color millions of pixels of image in few 
micro seconds.
Since the 2D or 3D graphic processing providing large opportunity in parallel
data processing, GPU hardware usually composed of hundreds of cores with thousands
of functional units in each core (a.k.a "thread block" [#Quantitative]_) in 
N-Vidia processors. 
Or tens of cores with tens thousands of functional units in each core for big 
cores architecture.

3D modeling
------------

Through creating 3D model with Triangles or Quads along on skin, the 3D model
is created with polygon mesh [#polygon]_ formed by all the vertices on the first image 
as follows,

.. _modeling1: 
.. figure:: ../Fig/gpu/modeling1.png
  :align: center
  :scale: 80 %

  Creating 3D model and texturing

After the next smooth shading [#polygon]_, the vertices and edge line are covered 
with color (or remove edges), then model looks much more smooth [#shading]_. 
Furthermore, after texturing (texture mapping), the model looks real more 
[#texturemapping]_.
 
To get to know how animation for a 3D modeling, please look video here [#animation1]_.
In this series of video, you find the 3D modeling tools creating Java instead of
C/C++ code calling OpenGL api and shaders. It's because Java can call OpenGL api
through a wrapper library [#joglwiki]_.

3D Rendering
------------

3D rendering is the process of converting 3D models into 2D images on a computer 
[#3drendering_wiki]_. The steps as the following Figure [#rendering]_.

.. _rendering_pipeline1: 
.. figure:: ../Fig/gpu/rendering_pipeline.png
  :align: center
  :scale: 80 %

  Diagram of the Rendering Pipeline. The blue boxes are programmable shader stages.


For 2D animation, the model is created by 2D only (1 face only), so it only can be 
viewed from the same face of model. If you want to display different faces of model,
multiple 2D models need to be created and switch these 2D models for face(flame) to 
face(flame) from time to time [#2danimation]_.

GLSL (GL Shader Language)
-------------------------

OpenGL is a standard for designing 2D/3D animation in computer graphic.
To do animation well, OpenGL provides a lots of api(functions) call for
graphic processing. The 3D model construction tools such as Maya, Blender, ...,
only need to call this api to finish the 3D to 2D projecting function in computer.
An OpenGL program looks like the following,

.. code-block:: c++

  Vertex shader
  
  #version 330 core
  layout (location = 0) in vec3 aPos; // the position variable has attribute position 0
    
  out vec4 vertexColor; // specify a color output to the fragment shader
  
  void main()
  {
      gl_Position = vec4(aPos, 1.0); // see how we directly give a vec3 to vec4's constructor
      vertexColor = vec4(0.5, 0.0, 0.0, 1.0); // set the output variable to a dark-red color
  }
  Fragment shader
  
  #version 330 core
  out vec4 FragColor;
    
  in vec4 vertexColor; // the input variable from the vertex shader (same name and same type)  
  
  void main()
  {
      FragColor = computeColorOfThisPixel(...);
  } 
  
  // openGl user program
  int 
  main(int argc, char **argv)
  {
  // init window, detect user input and do corresponding animation by calling opengl api
  }

The last main() is programed by user clearly. Let's explain what the first two 
main() work for. 
As you know, the OpenGL is a lots of api to let programmer display the 3D object 
into 2D computer screen explained from book concept of computer graphic.
3D graphic model can set light and object texture by user, next calculating the 
postion of each vertex and color for each pixel automatically by 3D software 
and GPU, finally display the color of each pixel in computer screen.
But in order to let user/programmer add some special effect or decoration in 
coordinate of each vertex or color of each pixel, OpenGL provides these two 
functions to do it. 
Programmer can add their converting functions then compiler translate them 
into GPU instructions running on GPU processor. With these two shaders, new 
features have been added to allow for increased flexibility in the rendering 
pipeline at the vertex and fragment level [#shaderswiki]_.
Unlike the shaders example here [#shadersex]_, some shaders converting function 
in vertex or color(Fragment shade) are more complicated according the scenes of 
animation. Here is an example [#glsleffect]_.
In wiki shading page [#shading]_, Gourand and Phong shading methods make the
surface of object more smooth are achieved by glsl. Example glsl code of Gourand 
and Phong shading on OpenGL api are here [#smoothshadingex]_.
Since the hardware of graphic card and software graphic driver can be changed, 
the compiler is run on-line which means compile the shaders program when it is 
run at first time.
The shaders program is C-like syntax and can be compiled in few mini-seconds, 
add up this few mini-seconds of on-line compile time in running OpenGL 
program is a good choice for dealing the cases of driver software or gpu 
hardware changes [#onlinecompile]_. 

OpenGL Shader compiler
-----------------------

OpenGL standard is here [#openglspec]_. The OpenGL is for desktop computer or server
while the OpenGL ES is for embedded system [#opengleswiki]_. Though shaders are only
a small part of the whole OpenGL software/hardware system. It is still a big effort 
to finish the compiler implementation since there are lots of api need to be 
implemented.
For example, the number of texture related api is close to one hundred for code
generation since they include different api names with different operands for 
each api name.
This implementation can be done by generating llvm extended intrinsic functions 
from shader parser of frontend compiler, and then llvm backend convert those intrinsic 
to gpu instructions as follows,

.. code-block:: console

  #version 320 es
  uniform sampler2D sampler_2d;
  out vec4 FragColor;
  
  void main()
  {
      FragColor = texture(sampler_2d, uv_2d, bias);
  }
  
  ...
  !1 = !{!"sampler_2d"}
  !2 = !{i32 INT_SAMPLER_2D} : INT_SAMPLER_2D is integer value for sampler2D, for example: 0x0f02
  ; A named metadata.
  !sampler_2d_var = !{!1, !2}

  define void @main() #0 {
      ...
      %1 = @llvm.gpu0.texture(metadata !sampler_2d_var, %1, %2, %3); // %1: %sampler_2d, %2: %uv_2d, %3: %bias
      ...
  }
  
  ...
     // gpu machine code
      sample2d_inst $1, $2, $3 // $1: %sampler_2d, $2: %uv_2d, $3: %bias
      
About llvm intrinsic extended function, please refer this book here [#intrinsiccpu0]_.

.. code-block:: c++

  gvec4 texture(gsampler2D sampler, vec2 P, [float bias]);


.. _sampling: 
.. figure:: ../Fig/gpu/sampling_diagram.png
  :align: center
  :scale: 60 %

  Relationships between the texturing concept [#textureobject]_.

The figure of relationships between the texturing concept as above.
The texture object is not bound directly into the shader (where the actual 
sampling takes place). Instead, it is bound to a 'texture unit' whose index 
is passed to the shader. So the shader reaches the texture object by going 
through the texture unit. There are usually multiple texture units available 
and the exact number depends on the capability of your graphis card [#textureobject]_. 
A texture unit, also called a texture mapping unit (TMU) or a texture processing 
unit (TPU), is a hardware component in a GPU that does sampling.
The argument sampler in texture function as above is sampler_2d index from
'teuxture unit' for texture object [#textureobject]_. 

'sampler uniform variable':

There is a group of special uniform variables for that, according to the texture 
target: 'sampler1D', 'sampler2D', 'sampler3D', 'samplerCube', etc. 
You can create as many 'sampler uniform variables' as you want and assign the 
value of a texture unit to each one from the application. 
Whenever you call a sampling function on a 'sampler uniform variable' the 
corresponding texture unit (and texture object) will be used [#textureobject]_.

In order to let the 'texture unit' binding by driver, frontend compiler must
pass the metadata of 'sampler uniform variable' (sampler_2d_var in this example) 
[#samplervar]_ to backend, and backend must 
allocate the metadata of 'sampler uniform variable' in the compiled 
binary file [#metadata]_. After gpu driver executing glsl on-line compiling,
driver read this metadata from compiled binary file and maintain a 
table of {name, SamplerType} for each 'sampler uniform variable'.

.. _sampling_binding: 
.. figure:: ../Fig/gpu/sampling_diagram_binding.png
  :align: center

  Binding sampler variables [#tpu]_.

As Figure: Binding sampler variables, the Java OpenGL wrapper api
gl.bindTexture binding 'Texture Object' to 'Texture Unit'. 
The gl.getUniformLocation and gl.uniform1i associate 'Texture Unit' to
'sampler uniform variables'. 

The gl.uniform1i(xLoc, 1) where 1 is 
'Texture Unit 1', 2 is 'Texture Unit 2', ..., etc [#tpu]_.

Api,

.. code-block:: c++

  x_texture_location = gl.getUniformLocation(pro, "x");
  
will get the location from the table for 'sampler uniform variable' x that
driver created as follows,

.. code-block:: console

  {"x", INT_SAMPLER_2D, x_location} : INT_SAMPLER_2D is integer value for Sampler2D type


Api,

.. code-block:: c++

  gl.uniform1i( x_texture_location, 1 );
  
will binding location(memory address) of 'sampler uniform variable' x to 
'Texture Unit 1' by writing 1 to the glsl binary metadata location of
'sampler uniform variable' x as follows,

.. code-block:: console

  {x_location, 1} : 1 is 'Texture Unit 1', x_location is the location(memory address) of 'sampler uniform variable' x
  
This api will set the descriptor register of gpu with this {x_location, 1} 
information.
  
Then, when executing the texture instructions from glsl binary file on gpu,

.. code-block:: console

  // gpu machine code
  sample2d_inst $1, $2, $3 // $1: %sampler_2d, $2: %uv_2d, $3: %bias
      
the corresponding 'Texture Unit 1' on gpu will be executing through 
descriptor register of gpu {x_location, 1} in this example.

The scenario comes from part of my imagination.

Since 'Texture Unit' is limited hardware accelerator on gpu, OpenGL
providing api to user program for binding 'Texture Unit' to 'Sampler Variables'
to doing load balance in using the 'Texture Unit'. With this mechanism, the 
compiled glsl binary is allowing to do load balance through OpenGL api without
recompiling glsl. The glsl on-line compiling only be triggered at first time of
running program. It is kept in cache and is executing directly after first 
time of compiling.
Fast texture sampling is one of the key requirements for good GPU performance 
[#tpu]_.

Even llvm intrinsic extended function providing an easy way to do code 
generation through llvm td (Target Description) file written, 
GPU backend compiler is still a little complex than CPU backend. 
(When counting in frontend compier such as clang or other toolchain such
as linker, JIT, gdb/lldb, of course, CPU compiler is much much complex than
GPU compiler)

Here is the software stack of 3D graphic system for OpenGL in linux [#mesawiki]_.

General purpose GPU
--------------------

Since GLSL shaders provide a general way for writing C code in them, if applying
a software frame work instead of OpenGL api, then the system can run some data
parallel computation on GPU for speeding up and even get CPU and GPU executing 
simultaneously. Or Any language that allows the code running on the CPU to poll 
a GPU shader for return values, can create a GPGPU framework [#gpgpuwiki]_.

The following is a CUDA example to run large data in array on GPU [#cudaex]_ 
as follows,

.. code-block:: c++

  __global__
  void saxpy(int n, float a, float *x, float *y)
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
  }
  
  int main(void)
  {
    ...
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    ...
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    ...
  }

The main() run on CPU while the saxpy() run on GPU. Through 
cudaMemcpyHostToDevice and cudaMemcpyDeviceToHost, CPU can pass data in x and y 
array to GPU and get result from GPU to y array. 
Since both of these memory transfer trigger the DMA functions without CPU operation,
it maybe speed up by running both CPU/GPU with their data in their own cache.
When the GPU function is dense computation in array such as MPEG4 encoder or
deep learning for tuning weights, it mays get much speed up [#mpeg4speedup]_. 
But when GPU function is matrix addition and CPU will idle for waiting 
GPU's result. It mays slow down than doing matrix addition by CPU only.
Arithmetic intensity is defined as the number of operations performed per word of 
memory transferred. It is important for GPGPU applications to have high arithmetic 
intensity else the memory access latency will limit computational speedup 
[#gpgpuwiki]_. 

Wiki here [#gpuspeedup]_ includes speepup applications for gpu as follows:

General Purpose Computing on GPU, has found its way into fields as diverse as 
machine learning, oil exploration, scientific image processing, linear algebra,
statistics, 3D reconstruction and even stock options pricing determination.
And section "GPU accelerated video decoding and encoding" for video compressing
more.


Vulkan and spir-v
-----------------

Though OpenGL api existed in higher level with many advantages from sections
above, sometimes it cannot compete in efficience with direct3D providing 
lower levels api for operating memory by user program [#vulkanapiwiki]_. 
Vulkan api is lower level C/C++ api to fill the gap allowing user program to 
do these things in OpenGL to compete against Microsoft direct3D. 
Here is an example [#vulkanex]_. Meanwhile glsl is C-like language. The vulkan 
infrastructure provides tool to compile glsl into an Intermediate Representation 
form (IR) called spir-v [#spirvtoolchain]_. 
As a result, it saves part of compiling time from glsl to gpu instructions on-line 
since spir-v is IR of level closing to llvm IR [#spirvwiki]_. 
In addition, vulkan api reduces gpu drivers efforts in optimization and code 
generation [#vulkanapiwiki]_. These standard provide user programmer option in 
using vulkan/spir-v or OpenGL/glsl, and allow them to pre-compile glsl into spir-v
to saving part of on-line compiling time.

With vulkan and spir-v standard, the gpu can be used in OpenCL for Parallel 
Programming of Heterogeneous Systems [#opencl]_ [#computekernelwiki]_.


.. [#Quantitative] Book Figure 4.13 of Computer Architecture: A Quantitative Approach 5th edition (The
       Morgan Kaufmann Series in Computer Architecture and Design)


.. [#polygon] https://www.quora.com/Which-one-is-better-for-3D-modeling-Quads-or-Tris


.. [#shading] https://en.wikipedia.org/wiki/Shading

.. [#texturemapping] https://en.wikipedia.org/wiki/Texture_mapping

.. [#animation1] https://www.youtube.com/watch?v=f3Cr8Yx3GGA

.. [#joglwiki] https://en.wikipedia.org/wiki/Java_OpenGL


.. [#3drendering_wiki] https://en.wikipedia.org/wiki/3D_rendering

.. [#rendering] https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview

.. [#2danimation] https://tw.video.search.yahoo.com/search/video?fr=yfp-search-sb&p=2d+animation#id=12&vid=46be09edf57b960ae79e9cd077eea1ea&action=view


.. [#shaderswiki] https://en.m.wikipedia.org/wiki/OpenGL_Shading_Language

.. [#shadersex] https://learnopengl.com/Getting-started/Shaders

.. [#glsleffect] https://www.youtube.com/watch?v=LyoSSoYyfVU at 5:25 from beginning: combine different textures.

.. [#smoothshadingex] https://github.com/ruange/Gouraud-Shading-and-Phong-Shading

.. [#onlinecompile] https://community.khronos.org/t/offline-glsl-compilation/61784

.. [#openglspec] https://www.khronos.org/registry/OpenGL-Refpages/

.. [#opengleswiki] https://en.wikipedia.org/wiki/OpenGL_ES

.. [#intrinsiccpu0] http://jonathan2251.github.io/lbd/funccall.html#add-specific-backend-intrinsic-function

.. [#textureobject] http://ogldev.atspace.co.uk/www/tutorial16/tutorial16.html

.. [#tpu] http://math.hws.edu/graphicsbook/c6/s4.html

.. [#metadata] This can be done by llvm metadata. http://llvm.org/docs/LangRef.html#namedmetadatastructure http://llvm.org/docs/LangRef.html#metadata

.. [#samplervar] The type of 'sampler uniform variable' called "sampler variables". http://math.hws.edu/graphicsbook/c6/s4.html

.. [#mesawiki] https://en.wikipedia.org/wiki/Mesa_(computer_graphics)


.. [#gpgpuwiki] https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units

.. [#cudaex] https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/

.. [#mpeg4speedup] https://www.manchestervideo.com/2016/06/11/speed-h-264-encoding-budget-gpu/

.. [#gpuspeedup] https://en.wikipedia.org/wiki/Graphics_processing_unit

.. [#vulkanapiwiki] Vulkan offers lower overhead, more direct control over the GPU, and lower CPU usage... By allowing shader pre-compilation, application initialization speed is improved... A Vulkan driver only needs to do GPU specific optimization and code generation, resulting in easier driver maintenance... https://en.wikipedia.org/wiki/Vulkan_(API)

.. [#vulkanex] https://github.com/SaschaWillems/Vulkan/blob/master/examples/triangle/triangle.cpp

.. [#spirvtoolchain] glslangValidator is the tool used to compile GLSL shaders into SPIR-V, Vulkan's shader format. https://vulkan.lunarg.com/doc/view/1.0.39.1/windows/spirv_toolchain.html

.. [#spirvwiki] SPIR 2.0: LLVM IR version 3.4. SPIR-V 1.X: 100% Khronos defined Round-trip lossless conversion to llvm.  https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation

.. [#opencl] https://www.khronos.org/opencl/

.. [#computekernelwiki] https://en.wikipedia.org/wiki/Compute_kernel
