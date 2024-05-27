import qi
import sys
import time
import os
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model

model = load_model('trained_Xception_updated_model.keras')

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
app = qi.Application(sys.argv, url="tcps://10.0.0.8:9503")
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
# start video recording (testing purposes only)

start_time = time.time()
end_time = time.time()

# counter for each prediction
predictions_dict = {
    'beard pulling': 0,
    'hair pulling': 0,
    'nail bitting': 0,
    'non-bfrb': 0
}

print('started')

option = 'y'

while option == 'y':

    option = input('Would you like for pepper to perform a prediction? (y/n) ')

    start_time = time.time()

    while time.time() - start_time < 10:

        # Get a camera image.
        # image[6] contains the image data passed as an array of ASCII chars.
        naoImage = video_service.getImageRemote(videoClient)

        if naoImage is not None:
            # changing image to a usable structure
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_bytes = bytes(bytearray(array))

            frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape(imageHeight, imageWidth, 3)
            # Changing image to use as input for the model
            im = Image.fromarray(frame)
            im = im.resize((299, 299))
            im_array = np.array(im)
            im_array = im_array.astype('float32')/255.0
            im_array = np.expand_dims(im_array, axis = 0)
            # performing prediction and extracting max prediction
            prediction = model.predict(im_array)
            predicted_classes = np.argmax(prediction, axis=1)
            # additional time check to prevent going over 10s
            if time.time() - start_time < 10:
                # check if behavior is not non-BFRB and confidence level is above 60%
                if predicted_classes[0] != 3 and np.max(prediction) > 0.6:
                    # translating prediction to behavior class
                    behavior = classes[predicted_classes[0]]
                    tts = app.session.service("ALTextToSpeech")

                    print(prediction)

                    # increasing counter for one behavior
                    predictions_dict[behavior] += 1
                    # only interact if behavior has been predicted more than once
                    if predictions_dict[behavior] > 1:
                        if behavior == 'hair pulling':
                            tts.say("You are pulling your hair")
                        elif behavior == 'beard pulling':
                            tts.say("You are pulling your beard")
                        elif behavior == 'nail bitting':
                            tts.say("You are biting your nails")
                        # resetting 10s timer
                        start_time = time.time()
                        predictions_dict = {
                            'beard pulling': 0,
                            'hair pulling': 0,
                            'nail bitting': 0,
                            'non-bfrb': 0
                        }
                        
                        break
                else:
                    # debugging purposes
                    print(classes[predicted_classes[0]])
                    behavior = 'non-bfrb'
            else:
                # warning that 10s time has passed, resetting time and counter
                print('%f second span passed' %(time.time()-start_time))
                start_time = time.time()
                predictions_dict = {
                        'beard pulling': 0,
                        'hair pulling': 0,
                        'nail bitting': 0,
                        'non-bfrb': 0
                    }
                break

        else:
            print('image not received')
            behavior = 'non-bfrb'
        
# ending script
video_service.unsubscribe(videoClient)

print('finished')

