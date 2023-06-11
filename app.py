import gradio as gr
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from generator import Generator
from discriminator import Discriminator
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(latent_dim=200, image_channels=3).to(device)
generator.load_state_dict(torch.load("generator.pth"))
# Define a function to generate faces
def generate_faces():
    num_faces = 24

    # Generate random noise
    noise = torch.randn(num_faces, 200, 1, 1, device=device)
    # Generate images using the generator model
    with torch.no_grad():
        generated_faces = generator(noise)
    
    # Convert the generated faces to PIL images
    generated_faces = make_grid(generated_faces.cpu(), nrow=8, normalize=True)
    pil_images = ToPILImage()(generated_faces)

    return pil_images

# Create a Gradio interface
title = "FaceForge"
description = "A Face Generating Project"
outputs = gr.outputs.Image(type="pil")
iface = gr.Interface(fn=generate_faces, inputs=None, outputs=outputs, title=title, description=description, allow_flagging="never")

# Define a function to update the output with generated faces
def update_output():
    iface.outputs = [generate_faces()]

# Add a button to generate faces
iface.launch(share=True, inline=False)