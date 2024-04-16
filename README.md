Needs to be ran on Linux or a Linux VM. Ubuntu is the verified method.

Python 3 is required

Libraries:

- qi -> "pip install qi"
- PIL -> "pip install Pillow"
- cv2 -> "pip install opencv-python"
- numpy -> "pip install numpy"
- keras -> "pip install keras"
- tensorflow -> "pip install tensorflow"

Documentation for pepper AL commands:
http://doc.aldebaran.com/2-5/naoqi/index.html

Structure:

- data_cleaning.py -> Organizes files
- model_fitting.py -> Fine-tunes all models
- pepper_runner.py -> Script that connects and interacts with the Pepper Robot

Dataset:

Self-evaluation:

We believe we achieved a satisfactory system considering our initial proposal. The only thing we did not have time to implement was using a Large Language Model to interact with the user. We had, however, mentioned that this implementation would have been time dependent, and chose not to do it given the additional constraints that it would impose in our project. Apart from that, we implemented everything else laid out in our initial project proposal.
