import mnist_convnet as cnn
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

model = cnn.mnist_convnet(8, 5)
model.load_state_dict(torch.load("./mnist_feedforward.pth"))
# Create a new Tkinter window
window = tk.Tk()

# Create a new canvas to draw on
canvas = tk.Canvas(window, width=300, height=300, bg='white')
canvas.pack()

# Create a new PIL Image object to draw on and an ImageDraw object
image = Image.new("RGB", (300, 300), "white")
draw = ImageDraw.Draw(image)
prediction_made = False

def save_image():
    global prediction_made
    # Resize and convert image to grayscale
    image_gray = image.convert("L").resize((28, 28))
    
    # Invert image colors (to match MNIST)
    image_gray = ImageOps.invert(image_gray)
    
    # Convert image data to a PyTorch tensor
    tensor = torch.from_numpy(np.array(image_gray)).view(1, 784)
    
    # Normalize tensor to match MNIST
    tensor = tensor.float() / 255
    
    prediction = model.predict_input(tensor)
    canvas.delete("all")
    draw.rectangle([0, 0, 500, 500], fill="white")
    canvas.create_text(150, 150, text=prediction, font=("Purisa", 150), fill="blue")
    prediction_made = True

def clear_canvas(event):
    # Clear the canvas
    global prediction_made
 
    if prediction_made:
        # Clear the canvas and the image
        canvas.delete("all")
        draw.rectangle([0, 0, 500, 500], fill="white")
        # Reset the flag
        prediction_made = False

def draw_points(event):
    # Draw on the PIL Image
    draw.ellipse([event.x-5, event.y-5, event.x+5, event.y+5], fill="black")
    
    # Draw on the Tkinter Canvas
    canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill="black", outline="")

# Bind the drawing function to the Tkinter Canvas
canvas.bind("<B1-Motion>", draw_points)
canvas.bind("<1>", clear_canvas)
# Add a button to save the image
button = tk.Button(window, text="Save", command=save_image)
button.pack()

# Run the Tkinter window
window.mainloop()





