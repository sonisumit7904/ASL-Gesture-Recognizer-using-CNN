June,2023 - July,2023

ASL (American Sign Language) Hand gesture recognizer using CNN (Convolutional Neural Network) - I trained a model using online available ASL model dataset (2000 images for each 
letter) , The model was trained to recognize 26 english alphabet letters.

Used 2 levels of prediction algorithm - because model was getting confused for some group of letters.

# To Install and Run the Project

1. install python 3.6 or above.

2. install pip if not already present.

3. Click Windows+R and type cmd

4. In cmd, install the following external library required to run the project - 

```
python -m pip install tensorflow --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
pip install tensorflow==2.0.0-alpha0
pip install keras (if keras doesnt work just replace keras.model to tensorflow.keras.model)
pip install pillow      
pip install numpy
pip install keras
```

5. start HandGestureMainFinal.py