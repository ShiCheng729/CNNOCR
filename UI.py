from Tkinter import *
import cv2
import char
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


global mask
canvas_width = 300
canvas_height = 500
mask = np.ones((300,500), np.uint8)*255
def paint( event ):
  python_color = "#000000"
  x1, y1 = ( event.x - 30 ), ( event.y - 30 )
  x2, y2 = ( event.x + 30 ), ( event.y + 30 )
  #coor.append([event.x,event.y])
  #mask[300-event.x,event.y] = 0
  cv2.circle(mask,(event.y,300-event.x), 30, (0,0,255), -1)
  w.create_oval( x1, y1, x2, y2, fill = python_color )

def callback():
  image = np.rot90(mask,axes=(1,0))
  result,p = char.image_proccessing(image)




def clear():
  global mask
  mask = np.ones((300,500), np.uint8)*255
  w.delete("all")




master = Tk()
master.title( "Painting using Ovals" )
w = Canvas(master, width=300, height=500)
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )

b = Button(master, text="OK", command=callback)
b.pack()
b2 = Button(master,text='Clear',command=clear)
b2.pack()
message = Label( master, text = "Please draw a character" )
message.pack( side = BOTTOM )

mainloop()