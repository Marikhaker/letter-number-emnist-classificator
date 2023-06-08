# General developing info:
From the beginning i started to work in Google Colab, using GPU. Then, when model was trained, most of the code was ready, i moved to Pycharm to develop.

# Train.py explanation:
## Dataset
**Loading dataset can take few minutes**

I used EMNIST (Extended MNIST) Balanced dataset to train my model. It looks like MNIST, but has also letters along with digits. Characters are white on black and is grayscaled. Balanced part of it has almost equal number of images for each class so i chose it to train on. Balanced has **47 classes** which are: 

```
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T']
```
It has following amount of data:
```
train: 112,800
test: 18,800
total: 131,600
```
Read more about it:
https://www.nist.gov/itl/products-and-services/emnist-dataset
https://www.kaggle.com/datasets/crawford/emnist

### Preprocessing

I loaded dataset form Tensorflow datasets datebase using tfds.load. It was already divided into Train and Test. I divided Train part to **85% for Train** and **15% for Validation**. At the  beginning, used only 25% of data from Test dataset, because 18k was taking long time to process and i dicided that 4.7k would be enough for this task. But then **evaluated for all 18.8k test images.**

Then i transposed images to be friendly for human to recognize. 
Then i make dataloaders with *tensorflow tfds* with **batch_size=256** and shuffling data. Test_dataloader is not shuffled and has **batch_size=1**, because batching made getting classification report when evaluating model with Test dataset harder.

I also save 25 example images in *"test_data_example"* folder

## Training
### Setup model
I created a model based on **LeNet-5**, but activations are ReLU, changed kernel_sizes for convolution layers and nums of convolutions, added batch_normalization. Structure of it you can see in the code and as image in project folder. It takes input_shape of **(batch_size, 28, 28, 1)**, has **2 conv layers**. **1st layer has 12 convolutions and kernel_size=12, 2nd layer has 24 convolutions and kernel_size=5**. I also added Dropout to deal with overfitting. Then **flatten and 2 dense layers** followed with batch_norm and dropout. The **final layer has len of 47** - number of classes in dataset, activation - softmax because of multiclass classification. 

Model has: **Total params: 284,335**

Model returns a list with length=47, numbers in each element are float from 0 to 1. Highest number, which could be extracted with *int(np.argmax)* is a prediction and means index of predicted class in LABELS list.

Optimizer is Adam, learning_rate=0.001, metrics="accuracy", loss="sparse_categorical_crossentropy". 

### Train
I added few callbacks: **EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.** 
To stop if model is overfitting, save best model each epoch and reduce learning rate if loss is not going down.

**Trained for 100 epoches, batch_size=256, early_stopping patience=10 epochs.**
After 60 epoches i've got early stopping: 
```
loss: 0.2876 - accuracy: 0.8911 - val_loss: 0.2932 - val_accuracy: 0.8917
```
**Test accuracy on 18.8k Test dataset: 0.88707**

I think accuracy could be increased up to ~0.92 with more epochs and some more changes made in Empirical way.

I also plot train/val accuracy/loss graphs.

### Training curves
<div align="center">
  <p>
     <img width="70%" src="https://github.com/Marikhaker/letter-number-emnist-classificator/blob/main/train_val%20acc_loss%20curves.jpg?raw=true">
  </p>
</div>

# Usage of inference program:
Program takes an input of string with target foldername. Than The output format is a single text line for every image in input directory “[character ASCII index in decimalformat], [POSIX path to image sample]”. The number of output lines is equal to images in input folder. **Output letters are ALL uppercase**. Programm can process different resolutions, they are being automatically reduced to size needed for model.
It can process following file extensions:

```
".jpg", ".jpeg", ".png", ".JPG", ".PNG", "JPEG"
```

Files are processed in alphabetical order.
**Tensorflow warnings was hidden to exclude unnecessary for this task output**

**It is prefarable to use absolute filepath for inference.py and test_data folder in CLI interface.**

**Filepaths must not include cyrillic words**

On my Linux machine, after i enter venv and install requirements, i can run inference script from console by typing following command into console, when files from "markiian_postavka_app" are extracted to uppermost directory:
```
python3 /app/inference.py --input /mnt/test_data
```
On Windows, with cmd opened in disc C root i can run script py using following command:
```
python C:\app\inference.py --input C:\mnt\test_data
```
## train.py usage
If you run train.py without changing anything, it wil load dataset - takes few minutes, and trai for 3 epoches. After 3 epoches training you could expect to get test accuracy ~ 0.86. 

**Dont forget that pretrained model will be overwritten if you run train.py.**
