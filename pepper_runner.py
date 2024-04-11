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
# print("started")

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

predictions_dict = {
    'beard pulling': 0,
    'hair pulling': 0,
    'nail bitting': 0,
    'non-bfrb': 0
}

print('started')

tts = app.session.service("ALTextToSpeech")
tts.say("Dedicas lbaba o mama o nizar lhbil, tahiya likom mn canada sfu")

option = 'y'

while option == 'y':

    option = input('Would you like for pepper to perform a prediction? (y/n) ')

    while True:

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
                tts = app.session.service("ALTextToSpeech")

                print(prediction)

                predictions_dict[behavior] += 1

                if predictions_dict[behavior] > 2:
                    if behavior == 'hair pulling':
                        tts.say("You're your hair")
                    elif behavior == 'beard pulling':
                        tts.say("You're pulling your beard")
                    elif behavior == 'nail bitting':
                        tts.say("You're biting your nails")
                    else:
                        tts.say("you're fine")
                    predictions_dict = {
                        'beard pulling': 0,
                        'hair pulling': 0,
                        'nail bitting': 0,
                        'non-bfrb': 0
                    }
                    
                    break
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

