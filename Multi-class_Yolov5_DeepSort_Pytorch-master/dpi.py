from PIL import Image

im = Image.open("photosDPI/Velocity_comparison.jpg")
im.save("photosDPI/jpg/Velocity_comparison.jpg", dpi=(600,600))