import keras
from PIL import Image, ImageOps
import numpy as np


def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Resize image
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Convert to array
    image_array = np.asarray(image).astype(np.float32)

    # Normalize
    normalized_image_array = (image_array / 127.0) - 1.0

    # Add batch dimension
    data = np.expand_dims(normalized_image_array, axis=0)

    # Run prediction
    prediction = model.predict(data)[0]  # shape = (3,)

    # Highest probability index
    label = np.argmax(prediction)

    return label, prediction  # return BOTH label and all probability scores
