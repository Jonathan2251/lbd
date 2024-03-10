//////////////////////////////////////////////////////////////////////////////
//
//  Triangles.cpp
//
//////////////////////////////////////////////////////////////////////////////

#include "vgl.h"
#include "LoadShaders.h"

enum VAO_IDs { Triangles, NumVAOs };
enum Buffer_IDs { ArrayBuffer, NumBuffers };
enum Attrib_IDs { vPosition = 0 };

GLuint  VAOs[NumVAOs];
GLuint  Buffers[NumBuffers];

const GLuint  NumVertices = 6;

//----------------------------------------------------------------------------
//
// init
//

void
init( void )
{
    glGenVertexArrays( NumVAOs, VAOs ); // Same with glCreateVertexArray( NumVAOs, VAOs ); 
      // https://stackoverflow.com/questions/24441430/glgen-vs-glcreate-naming-convention
    // Make the new VAO:VAOs[Triangles] active, creating it if necessary.
    glBindVertexArray( VAOs[Triangles] );
    // opengl->current_array_buffer = VAOs[Triangles]
    
    GLfloat  vertices[NumVertices][2] = {
        { -0.90f, -0.90f }, {  0.85f, -0.90f }, { -0.90f,  0.85f },  // Triangle 1
        {  0.90f, -0.85f }, {  0.90f,  0.90f }, { -0.85f,  0.90f }   // Triangle 2
    };

    glCreateBuffers( NumBuffers, Buffers );
    
    // Make the buffer the active array buffer.
    glBindBuffer( GL_ARRAY_BUFFER, Buffers[ArrayBuffer] );
    // Attach the active VBO:Buffers[ArrayBuffer] to VAOs[Triangles]
    // as an array of vectors with 4 floats each.
    // Kind of like:
    // opengl->current_vertex_array->attributes[attr] = {
    //     type = GL_FLOAT,
    //     size = 4,
    //     data = opengl->current_array_buffer
    // }
    // Can be replaced with glVertexArrayVertexBuffer(VAOs[Triangles], Triangles, 
    // buffer[ArrayBuffer], ArrayBuffer, sizeof(vmath::vec2));, glVertexArrayAttribFormat(), ...
    // in OpenGL 4.5.
    
    glBufferStorage( GL_ARRAY_BUFFER, sizeof(vertices), vertices, 0);

    ShaderInfo  shaders[] =
    {
        { GL_VERTEX_SHADER, "media/shaders/triangles/triangles.vert" },
        { GL_FRAGMENT_SHADER, "media/shaders/triangles/triangles.frag" },
        { GL_NONE, NULL }
    };

    GLuint program = LoadShaders( shaders );
    glUseProgram( program );

    glVertexAttribPointer( vPosition, 2, GL_FLOAT,
                           GL_FALSE, 0, BUFFER_OFFSET(0) );
    glEnableVertexAttribArray( vPosition );
    // Above two functions specify vPosition to vertex shader at layout (location = 0)
}

//----------------------------------------------------------------------------
//
// display
//

void
display( void )
{
    static const float black[] = { 0.0f, 0.0f, 0.0f, 0.0f };

    glClearBufferfv(GL_COLOR, 0, black);

    glBindVertexArray( VAOs[Triangles] );
    glDrawArrays( GL_TRIANGLES, 0, NumVertices );
}

//----------------------------------------------------------------------------
//
// main
//

#ifdef _WIN32
int CALLBACK WinMain(
  _In_ HINSTANCE hInstance,
  _In_ HINSTANCE hPrevInstance,
  _In_ LPSTR     lpCmdLine,
  _In_ int       nCmdShow
)
#else
int
main( int argc, char** argv )
#endif
{
    glfwInit();

    GLFWwindow* window = glfwCreateWindow(800, 600, "Triangles", NULL, NULL);

    glfwMakeContextCurrent(window);
    gl3wInit();

    init();

    while (!glfwWindowShouldClose(window))
    {
        display();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();
}
