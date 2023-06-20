import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

w,h= 500,500
def square():
    glBegin(GL_QUADS)
    glVertex2f(100.0, 100.0)
    glVertex2f(200.0, 100.0)
    glVertex2f(200.0, 200.0)
    glVertex2f(100.0, 200.0)
    glEnd()

def iterate():
    glViewport(0, 0, 500, 500)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, 500.0, 0.0, 500.0, 0.0, 1.0)
    glMatrixMode (GL_MODELVIEW)
    glLoadIdentity()

def showScreen():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    iterate()
    glColor3f(1.0, 0.0, 3.0)
    square()
    glutSwapBuffers()

    # Get the dimensions of the frame buffer
    width, height = 500, 500

    # Read the pixels from the frame buffer
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # Convert the pixels into a numpy array
    image = np.frombuffer(pixels, np.uint8).reshape(height, width, 3)

    # Flip the image vertically (OpenGL's origin is at the bottom left)
    image = np.flipud(image)

    # Create a PIL image and save it to a file
    Image.fromarray(image).save('output.png')

def timer(x):
    glutLeaveMainLoop()

glutInit()
glutInitDisplayMode(GLUT_RGBA)
glutInitWindowSize(500, 500)
glutInitWindowPosition(0, 0)
window = glutCreateWindow("noob-tutorials.com")
glutDisplayFunc(showScreen)
glutIdleFunc(showScreen)
glutTimerFunc(20 * 1000, timer, 0) # stop after 20 seconds
glutMainLoop()
