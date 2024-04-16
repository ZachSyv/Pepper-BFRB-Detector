# AI powered BFRB detection on pepper
This repostiory is created to showcase the code behind the group 18 project for CMPT 419 Spring 2024 project - AI powered BFRB detection on pepper.


## Repository structure
main/  
│   
├── README.md  
├── data_cleaning.py  
├── model_fitting.py  
├── model_testing.py   
└── pepper_runner.py  

## Requirements:
To run this code, you need to have access to a Linux based system or a Linux VM with Python 3 installed. In addition to that you need following python packages:
- qi -> "pip install qi"
- PIL -> "pip install Pillow"
- cv2 -> "pip install opencv-python"
- numpy -> "pip install numpy"
- keras -> "pip install keras"
- tensorflow -> "pip install tensorflow"

Documentation for pepper AL commands:
http://doc.aldebaran.com/2-5/naoqi/index.html


## Dataset:
The dataset collected for this project is provided as a zip file. 

## Data Preprocessing
For this part of the code, you need to run the file *data_cleaning.py*. This part of the code assumes that the dataset provided has the following structure:

  dataset/
  │   
  ├── images  
      ├── Beard-Pulling  
      ├── Hair-Pulling  
      ├── Nail-Biting   
      └── Non-BFRB

The code outputs training, testing, and validation splits. 60% of the original data goes toward training, 20% goes toward validation, and 20% goes toward testing.
These are randomly assigned. After running the *data_cleaning.py* file, your dataset directory should look like this:

  dataset/
  │   
  ├── images    
      ├── Beard-Pulling    
      ├── Hair-Pulling    
      ├── Nail-Biting     
      └── Non-BFRB  
  ├── test      
      ├── Beard-Pulling    
      ├── Hair-Pulling    
      ├── Nail-Biting     
      └── Non-BFRB  
  ├── train      
      ├── Beard-Pulling    
      ├── Hair-Pulling    
      ├── Nail-Biting     
      └── Non-BFRB  
  ├── validation      
      ├── Beard-Pulling    
      ├── Hair-Pulling    
      ├── Nail-Biting     
      └── Non-BFRB  

## Model Fitting
This part of the project requires you to run *model_fitting.py*. This files showcases all the pre-trained CNN models we fine-tuned and experimented with for this project. The code for data augmentation and fine-tuning is in this file. For each model, how the training loss, training accuracy, validation loss, and validation accuracy changes over the training period is recorded is provided as a png file.  

## Model Testing
This part of the project requires you to run *model_testing.py*. This file tests all the fine-tuned models on the test dataset created in the data preprocessing part. For each model, you will get a classification report and confusion matrix as an output. Please note that model training depends on random initialization, so the results you get varies slightly from run to run. 
  
## Self-evaluation:

We believe we achieved a satisfactory system considering our initial proposal. The only thing we did not have time to implement was using a Large Language Model to interact with the user. We had, however, mentioned that this implementation would have been time dependent, and chose not to do it given the additional constraints that it would impose in our project. Apart from that, we implemented everything else laid out in our initial project proposal.
