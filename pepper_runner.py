import qi # pip install qi is needed to import this library
import sys
import time
import os
from PIL import Image
import cv2
import numpy as np

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
app = qi.Application(sys.argv, url="tcps://10.0.0.4:9503")
logins = ("nao", "nao")
factory = AuthenticatorFactory(*logins)
app.session.setClientAuthenticatorFactory(factory)
app.start()
print("started")

#send camera information to local machine

video_service = app.session.service("ALVideoDevice")
resolution = 2    # VGA
colorSpace = 11   # RGB
fps = 5

videoClient = video_service.subscribeCamera("python_client", 0, resolution, colorSpace, fps)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (640, 480))

start_time = time.time()
duration = 10  # seconds
end_time = time.time()

while end_time - start_time < duration:

    # Get a camera image.
    # image[6] contains the image data passed as an array of ASCII chars.
    naoImage = video_service.getImageRemote(videoClient)

    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    image_bytes = bytes(bytearray(array))

    frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape(imageHeight, imageWidth, 3)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
   
    out.write(frame)
    end_time = time.time()
    print("Current delay", end_time - start_time)

video_service.unsubscribe(videoClient)
out.release()

print('finished recording')

#recive the behaviour detected from the local machine

# behaviour_detected = "nailbiting"

# if (behaviour_detected == "nailbiting"):
#     tts = app.session.service("ALTextToSpeech")
#     tts.say("Please stop biting your nails")
