from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = 'data/train'
val_dir = 'data/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range= 10,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    zoom_range=0.2,
    horizontal_flip= True
    )

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)

# Early Stopping: Eğitim sürecinde doğrulama hatası belirli bir sayıda epoch boyunca iyileşmediğinde durdurur
early_stopping = EarlyStopping(
    monitor='val_loss',  # İzlenecek metrik
    patience=10,         # İyileşme olmadan bekleyeceği epoch sayısı
    restore_best_weights=True  # En iyi ağırlıkları geri yükle
)

# Learning Rate Scheduler: Öğrenme oranını dinamik olarak azaltır
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # İzlenecek metrik
    factor=0.5,          # Öğrenme oranını azaltma faktörü
    patience=5,          # Öğrenme oranını azaltmadan önce bekleyeceği epoch sayısı
    min_lr=1e-6          # Minimum öğrenme oranı
)

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch = 28709 // 64,
    epochs=100,
    validation_data = val_generator,
    validation_steps = 7178 // 64,
    callbacks=[early_stopping, reduce_lr]  
)

# Eğitim sırasında doğruluk bilgilerini yazdırmak
print("Training Accuracy:", emotion_model_info.history['accuracy'])
print("Validation Accuracy:", emotion_model_info.history['val_accuracy'])

# Ortalama doğruluk oranları
mean_training_accuracy = sum(emotion_model_info.history['accuracy']) / len(emotion_model_info.history['accuracy'])
mean_validation_accuracy = sum(emotion_model_info.history['val_accuracy']) / len(emotion_model_info.history['val_accuracy'])

print(f"Ortalama Eğitim Doğruluğu: {mean_training_accuracy:.2f}")
print(f"Ortalama Doğrulama Doğruluğu: {mean_validation_accuracy:.2f}")

# En iyi epoch
best_epoch = emotion_model_info.history['val_accuracy'].index(max(emotion_model_info.history['val_accuracy'])) + 1
print(f"En iyi epoch: {best_epoch}. Epoch")

# Eğitim ve doğrulama doğruluğunu görselleştirme
plt.plot(emotion_model_info.history['accuracy'], label='Training Accuracy')
plt.plot(emotion_model_info.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

emotion_model.save_weights('model.h5')