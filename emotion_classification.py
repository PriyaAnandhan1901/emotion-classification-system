import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ─── Configuration ──────────────────────────────────────────────────────────
IMG_SIZE    = 48
NUM_CLASSES = 7
BATCH_SIZE  = 64
EPOCHS      = 50
DATA_DIR    = "data/FER-2013"   # expects data/FER-2013/train and data/FER-2013/test

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ─── Build CNN Model ─────────────────────────────────────────────────────────
def build_model():
    model = Sequential([
        # Block 1
        Conv2D(64, (3,3), padding="same", activation="relu",
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(64, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(256, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Fully Connected
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Data Generators ─────────────────────────────────────────────────────────
def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        f"{DATA_DIR}/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        f"{DATA_DIR}/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return train_gen, val_gen


# ─── Train ───────────────────────────────────────────────────────────────────
def train():
    print("[INFO] Building model ...")
    model = build_model()
    model.summary()

    train_gen, val_gen = get_data_generators()

    callbacks = [
        ModelCheckpoint("emotion_model.h5", monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
    ]

    print("[INFO] Training ...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print(f"[INFO] Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    return model, history


# ─── Real-Time Webcam Inference ───────────────────────────────────────────────
def run_webcam(model_path="emotion_model.h5"):
    print("[INFO] Loading model ...")
    model = tf.keras.models.load_model(model_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                              minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))   # (1, 48, 48, 1)

            preds   = model.predict(roi, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(preds)]
            conf    = preds.max() * 100

            label = f"{emotion} ({conf:.0f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x, y-th-12), (x+tw+6, y), (0, 255, 100), -1)
            cv2.putText(frame, label, (x+3, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        cv2.imshow("Emotion Classification  |  Press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        run_webcam()
