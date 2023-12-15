import numpy as np
from PIL import Image
from io import BytesIO

def load_image_into_numpy_array(file_contents):
    # Load the image using PIL
    image = Image.open(BytesIO(file_contents))
    
    # Resize the image to (150, 150)
    image = image.resize((150, 150))
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array
