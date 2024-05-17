
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import  Flatten
import sklearn
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
from keras.applications.densenet import DenseNet169, DenseNet121, DenseNet201
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.applications.xception import Xception
from keras.applications.convnext import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
import seaborn as sns
import gc

# Load numpy sets
x_train = np.load('/home/matheus_levy/workspace/crops_npy/x_train.npy')
y_train = np.load('/home/matheus_levy/workspace/crops_npy/y_train.npy')

x_val = np.load('/home/matheus_levy/workspace/crops_npy/x_val.npy')
y_val = np.load('/home/matheus_levy/workspace/crops_npy/y_val.npy')

x_test = np.load('/home/matheus_levy/workspace/crops_npy/x_test.npy')
y_test = np.load('/home/matheus_levy/workspace/crops_npy/y_test.npy')

# Ajust class from 1 to 0
y_train-=1
y_val-=1
y_test-=1

# Data Augmentation
transform = A.Compose([
    A.augmentations.geometric.rotate.Rotate(limit=15,p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.Blur(p=0.3),
    A.GaussNoise(var_limit=(0.1,0.3),p=0.4),
    A.Downscale(p=0.3),
])

def albumentations_augmentation(image):
    image = image/255.0
    augmented = transform(image=image)
    image = augmented['image'] * 255
    return image

custom_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=albumentations_augmentation
)

base_datagen = ImageDataGenerator(rescale=1./255)

# Batch_size
batch_size = 32  
epochs = 30
lr = 0.0001
seed = 123456  

# Generators 
train_generator = custom_datagen.flow(x_train, y_train, batch_size=batch_size, seed=seed)
val_generator = base_datagen.flow(x_val, y_val, batch_size=batch_size, seed=seed)

# Aux functions
def draw_confusion_matrix(true,preds):
    conf_matx = confusion_matrix(true, preds)
    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")
    plt.show()
    print(sklearn.metrics.classification_report(true, preds, digits=5))

def test_metrics(model, x,y_true):
  y_pred = modelpred(model,x) # Executa o predict
  draw_confusion_matrix(y_true, y_pred)
  return f1_score(y_true, y_pred, average='weighted')

def modelpred(model,test_full):
  y_get = model.predict(test_full)
  result = []

  for p in y_get:
    result.append(np.argmax(p))

  return result

# Function to train backbone
def search_backbone(backbone):
  print(backbone)
  if backbone == 'densenet121':
    model_backbone = DenseNet121(weights='imagenet',
                                include_top=False,
                                input_shape=(128,128, 3))

  elif backbone == 'densenet169':
    model_backbone = DenseNet169(weights='imagenet',
                                include_top=False,
                                input_shape=(128,128, 3))

  elif backbone == 'densenet201':
    model_backbone = DenseNet201(weights='imagenet',
                                include_top=False,
                                input_shape=(128,128, 3))

  elif backbone == 'efficientnetb0':
    model_backbone = EfficientNetB0(weights='imagenet',
                                    include_top=False,
                                    input_shape=(128,128, 3))

  elif backbone == 'efficientnetb1':
    model_backbone = EfficientNetB1(weights='imagenet',
                                    include_top=False,
                                    input_shape=(128,128, 3))

  elif backbone == 'efficientnetb2':
    model_backbone = EfficientNetB2(weights='imagenet',
                                    include_top=False,
                                    input_shape=(128,128, 3))

  elif backbone == 'efficientnetb3':
    model_backbone = EfficientNetB3(weights='imagenet',
                                    include_top=False,
                                    input_shape=(128,128, 3))

  elif backbone == 'efficientnetb4':
    model_backbone = EfficientNetB4(weights='imagenet',
                                    include_top=False,
                                    input_shape=(128,128, 3))

  elif backbone == 'efficientnetb5':
    model_backbone = EfficientNetB5(weights='imagenet',
                                    include_top=False,
                                    input_shape=(128,128, 3))

  elif backbone == 'efficientnetb6':
    model_backbone = EfficientNetB6(weights='imagenet',
                                    include_top=False,
                                    input_shape=(128,128, 3))

  elif backbone == 'efficientnetb7':
    model_backbone = EfficientNetB7(weights='imagenet',
                                    include_top=False,
                                    input_shape=(128,128, 3))

  elif backbone == 'xception':
    model_backbone = Xception(weights='imagenet',
                              include_top=False,
                              input_shape=(128,128, 3))

  elif backbone == 'InceptionV3':
    model_backbone = InceptionV3(weights='imagenet',
                                include_top=False,
                                input_shape=(128,128,3)
                                )

  elif backbone == 'ResNet50':
    model_backbone = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(128,128,3)
                              )

  elif backbone == 'ResNet101':
    model_backbone = ResNet101(weights='imagenet',
                              include_top=False,
                              input_shape=(128,128,3)
                              )
                  
  elif backbone == 'ResNet152':
    model_backbone = ResNet152(weights='imagenet',
                              include_top=False,
                              input_shape=(128,128,3)
                              )

  model_backbone.trainable = True

  base_model_output = model_backbone.output
  x = keras.layers.GlobalAveragePooling2D()(base_model_output)
  output = keras.layers.Dense(units=11, activation='softmax')(x)
  model = keras.models.Model(inputs=model_backbone.input, outputs=output)

  adam = Adam(learning_rate=lr)
  es = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                      mode='max',
                                      verbose=1,
                                      patience=10,
                                      min_delta=1e-3,
                                      restore_best_weights=True)

  reducelr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=3,
                              min_lr=0.000001,
                              mode='min')

  model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(train_generator,
                    epochs= epochs,
                    validation_data = val_generator,
                    shuffle= True,
                    batch_size= batch_size,
                    callbacks=[es,reducelr],
                    verbose=1)
  f1_test = test_metrics(model, x_test/255.0, y_test)
  f1_val = test_metrics(model, x_val/255.0, y_val)
  print(f'backbone: {backbone} f1_val: {f1_val} f1_test: {f1_test}')
  model.save(f'{backbone}.keras')
  del model
  gc.collect()
  return f1_val

backbone_list = ['densenet121', 'densenet169','densenet201', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3' , 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'xception', 'InceptionV3', 'ResNet50', 'ResNet101', 'ResNet152']
best_f1_score = 0
best_backbone = None

for backbone_name in backbone_list:
    try:
      f1_score_current = search_backbone(backbone_name)
    except Exception as e:
      print(f'erro: {e}')
      continue
    print(f'finished backbone: {backbone_name} with f1 {f1_score_current}')
    if f1_score_current > best_f1_score:
        best_f1_score = f1_score_current
        best_backbone = backbone_name

print('best result')
print(best_backbone)
print(best_f1_score)