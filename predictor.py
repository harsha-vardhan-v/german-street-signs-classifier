import tensorflow as tf
import numpy as np

def predict_with_model(model, img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (60, 60)) #(60, 60, 3)
    image = tf.expand_dims(image, axis=0) #(1, 60, 60, 3)

    predictions = model.predict(image)
    predictions = np.argmax(predictions)

    return predictions


if __name__ == '__main__':
    
    img_path = 'D:\\Projects\\Learning\\Deep Learning\\tf-image-processing\\german_street_signs\\Test\\8\\04373.png'
    model = tf.keras.models.load_model('./Models')

    prediction = predict_with_model(model, img_path)

    print(f'Prediction = {prediction}')