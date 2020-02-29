.. _sec-gpu:

GPU compiler
============

.. contents::
   :local:
   :depth: 4

Basicly CPU compiler is SISD (Single Instruction Single Data Architecture). 
The vector or multimedia instructions in CPU are small scaled of SIMD
(Single Instruction Multiple Data). GPU is a large scaled of SIMD processor
which need to color an millions pixel of image in few micro seconds.
Since the 2D or 3D graphic processing providing large opportunity in parallel
data processing, GPU hardware usually composed of hundreds of cores with thousands
of functional units (a.k.a "thread block" [#Quantitative]_) in each core that is 
small cores architecture in N-Vidia processors. 
Or tens of cores with tens thousands that is big cores architecture.

GLSL (GL Shader Language)
-------------------------

OpenGL is a standard for designing 2D/3D animation in computer graphic.
To do animation well, OpenGL provide a lots of api function calls for
graphic processing. The 3D model construction tools such as Maya, ...,
can call this api to finish the 3D to 2D drawing function in computer.
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

The last main() is programed by user clearly. Let's explain what the first two main()'s 
work for. As you know, the OpenGL is a lots of api functions to let programmer display 
the 3D object into 2D computer screen. As you know from book concept of computer graphic.
3D graphic model can set light and object texture and calculate the postion of each vertex
and color of each pixel, then display the color of each pixel in computer screen.
But in order to let programmer can add some special effect or decoration in coordinate of vertex or color
of pixel, OpenGL provide these two functions to do it. Programmer can add their converting 
functions and compiler translate them into GPU instructions running on GPU processor.
Not like the shaders example here [#shadersex]_, some shaders converting function in vertex 
or color(Fragment shade) are more complicated according the scene of animation timing.
Since the hardware of graphic card and software graphic driver can be changed, the compiler
is run on-line which means compile the shaders program when it is be run at first time.
The shaders program is C-like syntax and can be compiled in few mini-seconds [#onlinecompile]_. 




.. [#Quantitative] Book Figure 4.13 of Computer Architecture: A Quantitative Approach 5th edition (The
       Morgan Kaufmann Series in Computer Architecture and Design)

.. [#shadersex] https://learnopengl.com/Getting-started/Shaders

.. [#onlinecompile] https://community.khronos.org/t/offline-glsl-compilation/61784