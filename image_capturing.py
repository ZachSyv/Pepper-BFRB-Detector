import qi 
import sys
import time
import os
from PIL import Image
import cv2
import numpy as np
import glob


classes = ['Beard-Pulling', 'Hair-Pulling', 'Nail-Biting', 'Non-BFRB']

class Authenticator:

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def initialAuthData(self):
        return {'user': self.username, 'token': self.password}

class AuthenticatorFactory:

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def newAuthenticator(self):
        return Authenticator(self.username, self.password)

# Replace the URL with the IP of Pepper, get the ip from pressing the power button once
app = qi.Application(sys.argv, url="tcps://10.0.0.6:9503")
logins = ("nao", "nao")
factory = AuthenticatorFactory(*logins)
app.session.setClientAuthenticatorFactory(factory)
app.start()

#send camera information to local machine

video_service = app.session.service("ALVideoDevice")
resolution = 2    # VGA
colorSpace = 11   # RGB
fps = 5

videoClient = video_service.subscribeCamera("python_client", 0, resolution, colorSpace, fps)

start_time = time.time()
end_time = time.time()

autonomous_life = app.session.service('ALAutonomousLife')
autonomous_life.setState('solitary')


print('started')

option = 0

while option in [0, 1, 2, 3]:

    option = int(input('Which behavior would you like to record?\n0 - beard pulling\n1 - hair pulling\n2 - nail bitting\n3 - non-bfrb\n'))

    start_time = time.time()

    while time.time() - start_time < 30:

        # Get a camera image.
        # image[6] contains the image data passed as an array of ASCII chars.
        naoImage = video_service.getImageRemote(videoClient)

        if naoImage is not None:
            # changing image to a usable structure
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_bytes = bytearray(array)

            im = Image.frombytes('RGB', (imageWidth, imageHeight), image_bytes)
            count = len(glob.glob(f'./{classes[option]}/*'))
            im.save(f'./{classes[option]}/{count}.png', "PNG")


        else:
            print('image not received')
            behavior = 'non-bfrb'
        
# ending script
video_service.unsubscribe(videoClient)

print('finished recording')

