from my_utils import order_test_set, split_data, create_generators
from dl_models import streetsigns_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

if __name__ == '__main__':
    if False:
        path_to_data = 'D:\\Projects\\Learning\\Deep Learning\\tf-image-processing\\german_street_signs\\Train'
        train_path = 'D:\\Projects\\Learning\Deep Learning\\tf-image-processing\\gbrt_training_data\\Train'
        val_path = 'D:\\Projects\\Learning\\Deep Learning\\tf-image-processing\\gbrt_training_data\\Valid'

        split_data(path_to_data, train_path, val_path)

    if False:
        path_to_images = 'D:\\Projects\\Learning\\Deep Learning\\tf-image-processing\\german_street_signs\\Test'
        path_to_csv = 'D:\\Projects\\Learning\\Deep Learning\\tf-image-processing\\german_street_signs\\Test.csv'

        order_test_set(path_to_images, path_to_csv)

    train_path = 'D:\\Projects\\Learning\Deep Learning\\tf-image-processing\\gbrt_training_data\\Train'
    val_path = 'D:\\Projects\\Learning\\Deep Learning\\tf-image-processing\\gbrt_training_data\\Valid'
    test_path = 'D:\\Projects\\Learning\\Deep Learning\\tf-image-processing\\german_street_signs\\Test'
    batch_size = 64
    epochs = 15
    lr = 0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size, train_path, val_path, test_path)
    nbr_classes = train_generator.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:
        model_path = './Models'
        ckpt = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=3
        )

        model = streetsigns_model(nbr_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt, early_stopping]
        )

    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print('Validation set evaluation: ')
        model.evaluate(val_generator)
        print('Test set evaluation: ')
        model.evaluate(test_generator)

