from tensorflow.keras.models import load_model


def inspect_model(model_path):
    """Inspect the input shape and supported gestures of a Keras model."""
    model = load_model(model_path, compile=False)
    
    # check input shape
    input_shape = model.input_shape
    print(f"Expected input shape (image size): {input_shape[1:]}")

    # check output shape
    output_shape = model.output_shape
    num_classes = output_shape[1]
    print(f"Number of supported gestures: {num_classes}")

    # attempt to retrieve gesture labels (if present)
    try:
        class_names = model.class_names  # check if labels are stored as an attribute
        print("Supported gestures (labels):", class_names)
    except AttributeError:
        print("Gesture labels not directly stored in the model. \n" \
              "You may need to check your training script for the label mapping.")

if __name__ == "__main__":
    inspect_model("resources/model.h5")
