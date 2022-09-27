# tutorial from https://www.youtube.com/watch?v=7xc0Fs3fpCg&t=79s
# git repo from https://github.com/nicknochnack/StableDiffusionApp/blob/main/app.py
import tkinter as tk
import customtkinter as ctk 

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

#create app
app = tk.Tk()
app.geometry("532x622")
app.title("Stable Bud")
ctk.set_apperance_mode("dark")

app.mainloop()