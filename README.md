# X-RayImageClassification_DL

## Dataset
The Data, for this project, were collected from Kaggle. https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
ImageTest.zip file consists of some images that you can use them as input of my CNN model.

## Colab Notebook
You can see the whole Colab Notebook by [clicking here](https://nbviewer.org/github/icsd13152/X-RayImageClassification_DL/blob/main/X_RayImageClassification_With_Deep_Learning.ipynb)

## Requirements

In order to execute the Code, you must install the packages that their name and versions are located in requirements.txt file.

In order to fetch and install all of the packages you have to run the below command:
```
$ pip install -r requirements.txt

```
(it will take about 3-4 minutes. It depends on Network/Internet speed.)

## Model Architecture
Below the Architecture of deployed model.
![arch](https://github.com/icsd13152/X-RayImageClassification_DL/blob/main/mockup/model4.PNG?raw=true)

## Application
Below an image of use-case for the application.  
![mockup](https://github.com/icsd13152/X-RayImageClassification_DL/blob/main/mockup/mockup.PNG?raw=true)
There is also the appDemo.py file that is the application for predictions.

```
$ cd src/
$ python appDemo.py

```