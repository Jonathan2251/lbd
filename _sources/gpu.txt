.. _sec-gpu:

Appendix C: GPU compiler
========================

.. contents::
   :local:
   :depth: 4

Basicly CPU compiler is SISD (Single Instruction Single Data Architecture). 
The vector or multimedia instructions in CPU are small scaled of SIMD
(Single Instruction Multiple Data). GPU is a large scaled of SIMD processor
which need to color an millions pixel of image in few micro seconds.
Since the 2D or 3D graphic processing providing large opportunity in parallel
data processing, GPU hardware usually composed of hundreds of cores with thousands
of functional units (a.k.a "thread block" [#Quantitative]_) in N-Vidia processors. 
Or tens of cores with tens thousands that is big cores architecture.

3D modeling
------------

Through creating 3D model with Triangles or Quads, then set skin by texturing,
a 3D modeling can be created looks like the following Figure.

.. _modeling1: 
.. figure:: ../Fig/gpu/modeling1.png
  :align: center

  Creating 3D model and texturing
  
As above graph, setting the vertices along with skin form the polygon [#polygon]_ .
After shading, the vertices and edge line are covered with color (or remove edges), 
so the model looks much more smooth [#shading]_. Furthermore, after texturing 
(texture mapping), the model looks more real [#texturemapping]_.
 
  
To get to know how animation for a 3D modeling, please look video here [#animation1]_.

3D Rendering
------------

3D rendering is the 3D computer graphics process of converting 3D models into 2D 
images on a computer [#3drendering_wiki]_. The steps as the following Figure [#rendering]_.

.. _rendering_pipeline1: 
.. figure:: ../Fig/gpu/rendering_pipeline.png
  :align: center

  Diagram of the Rendering Pipeline. The blue boxes are programmable shader stages.


For 2D animation, the model is created by 2D only (1 side only), so it only can be 
viewed from the same face of model. If you want to display different side of model,
multiple 2D model need to be created and switch these 2D models from time to time.

GLSL (GL Shader Language)
-------------------------

OpenGL is a standard for designing 2D/3D animation in computer graphic.
To do animation well, OpenGL provide a lots of api(functions) call for
graphic processing. The 3D model construction tools such as Maya, ...,
can call this api to finish the 3D to 2D projecting function in computer.
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
main()'s work for. 
As you know, the OpenGL is a lots of api to let programmer display the 3D object 
into 2D computer screen explained from book concept of computer graphic.
3D graphic model can set light and object texture and calculate the postion of each vertex
and color of each pixel, then display the color of each pixel in computer screen.
But in order to let programmer add some special effect or decoration in 
coordinate of vertex or color of pixel, OpenGL provide these two functions to 
do it. Programmer can add their converting functions and compiler translate them 
into GPU instructions running on GPU processor. With these two shaders, new 
features have been added to allow for increased flexibility in the rendering 
pipeline at the vertex and fragment level [#shaderswiki]_.
Not like the shaders example here [#shadersex]_, some shaders converting function in vertex 
or color(Fragment shade) are more complicated according the scenes of animation.
Since the hardware of graphic card and software graphic driver can be changed, the compiler
is run on-line which means compile the shaders program when it is be run at first time.
The shaders program is C-like syntax and can be compiled in few mini-seconds [#onlinecompile]_. 
So, add up this on-line compile time in running OpenGL program is better to deal with
the driver or hardware of gpu changes. 

Shader compiler
---------------

OpenGL standard is here [#openglspec]_. The OpenGL is for desktop computer or server
while the OpenGL ES is for embedded system [#opengleswiki]_. Though shaders are only
a small part of the whole OpenGL software/hardware system. It is still a big effort 
to finish the compiler implementation since there are lots of api need to implement.
For example, the texture related api has close to one hundreds of api for code
generation include the api name and different operands in the same api name.
This implementation can done by generating llvm intrinsic function from shader's api
parser of frontend compiler, and designing llvm backend for those
extended llvm intrinsic functions to finish it as follows,

.. code-block:: c++
  
  #version 320 es
  out vec4 FragColor;
  
  void main()
  {
      FragColor = texture(sampler_2d, pos_2d, bias);
  }
  
  ...
  define void @main() #0 {
      ...
      %1 = @llvm.gpu0.texture(%sampler_2d, %pos_2d, %bias);
      ...
  }
  
  ...
     // gpu machine code
      sample2d_inst $1, $2, $3 // $1: %sampler_2d, $2: %pos_2d, $3: %bias
      
About llvm intrinsic extended function, please refer this book here [#intrinsiccpu0]_.

.. code-block:: c++

  gvec4 texture(gsampler2D sampler, vec2 P, [float bias]);


The texture object is not bound directly into the shader (where the actual 
sampling takes place). Instead, it is bound to a 'texture unit' whose index 
is passed to the shader. So the shader reaches the texture object by going 
through the texture unit. There are usually multiple texture units available 
and the exact number depends on the capability of your graphis card [#textureobject]_. 
A texture unit, also called a texture mapping unit (TMU) or a texture processing 
unit (TPU), is a hardware component in a GPU that does sampling.
Fast texture sampling is one of the key requirements for good GPU performance [#tpu]_.
So, the argument sampler in texture function as above is sampler_2d index. 
In order to let the 'texture unit' binding by driver, frontend compiler must
pass the name of 'texture unit' to backend, and backend must allocate the
(index, memaddr) of 'texture unit' in the compiled binary file.
Driver will be triggered and set memaddr when user program call api 
glGenTextures, glBindTexture and glTexImage2D before shader program
executing on gpu [#tpu]_.
So even llvm intrinsic extended function providing an easy way to do code 
generation through llvm td (Target Description) file written. 
GPU backend compiler is a little complex than CPU backend. 
    

.. [#Quantitative] Book Figure 4.13 of Computer Architecture: A Quantitative Approach 5th edition (The
       Morgan Kaufmann Series in Computer Architecture and Design)


.. [#polygon] https://en.wikipedia.org/wiki/Polygon_(computer_graphics)


.. [#shading] https://en.wikipedia.org/wiki/Shading

.. [#texturemapping] https://en.wikipedia.org/wiki/Texture_mapping

.. [#animation1] https://www.youtube.com/watch?v=f3Cr8Yx3GGA


.. [#3drendering_wiki] https://en.wikipedia.org/wiki/3D_rendering

.. [#rendering] https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview


.. [#shadersex] https://learnopengl.com/Getting-started/Shaders

.. [#shaderswiki] https://en.m.wikipedia.org/wiki/OpenGL_Shading_Language

.. [#onlinecompile] https://community.khronos.org/t/offline-glsl-compilation/61784

.. [#openglspec] https://www.khronos.org/registry/OpenGL-Refpages/

.. [#opengleswiki] https://en.wikipedia.org/wiki/OpenGL_ES

.. [#intrinsiccpu0] http://jonathan2251.github.io/lbd/funccall.html#add-specific-backend-intrinsic-function

.. [#textureobject] http://ogldev.atspace.co.uk/www/tutorial16/tutorial16.html

.. [#tpu] http://math.hws.edu/graphicsbook/c6/s4.html
