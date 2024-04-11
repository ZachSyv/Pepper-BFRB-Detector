import qi # pip install qi is needed to import this library
import sys
import time
import os
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model

model = load_model('trained_Xception_model.keras')

classes = ['beard pulling', 'hair pulling', 'nail bitting', 'non-bfrb']

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
duration = 60  # seconds
end_time = time.time()

print('started')

while end_time - start_time < duration:

    # Get a camera image.
    # image[6] contains the image data passed as an array of ASCII chars.
    naoImage = video_service.getImageRemote(videoClient)
    # print(time.time()-end_time)

    if naoImage is not None:

        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]
        image_bytes = bytes(bytearray(array))

        frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape(imageHeight, imageWidth, 3)

        im = Image.fromarray(frame)
        im = im.resize((299, 299))
        im_array = np.array(im)
        im_array = im_array.astype('float32')/255.0
        im_array = np.expand_dims(im_array, axis = 0)
        prediction = model.predict(im_array)
        predicted_classes = np.argmax(prediction, axis=1)

        if predicted_classes[0] != 3 and np.max(prediction) > 0.6:
            behavior = classes[predicted_classes[0]]
            time.sleep(5) # wait for 5 seconds to prevent multiple activations for the same behavior
            tts = app.session.service("ALTextToSpeech")
            if behavior == 'hair pulling':
                tts.say("Stop pulling your hair")
            elif behavior == 'beard pulling':
                tts.say("Stop pulling your beard")
            elif behavior == 'nail bitting':
                tts.say("Stop biting your nails")
        else:
            print(classes[predicted_classes[0]])
            behavior = 'non-bfrb'

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
    
        out.write(frame)
        end_time = time.time()
        print("Current delay", end_time - start_time)
    else:
        print('image not received')
        behavior = 'non-bfrb'
    

video_service.unsubscribe(videoClient)
out.release()

print('finished recording')


# else:
#     tts.say("you're fine")
