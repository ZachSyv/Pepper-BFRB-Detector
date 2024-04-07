import qi # pip install qi is needed to import this library
import sys
import time
from PIL import Image

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

videoClient = video_service.subscribeCamera("python_client", 0, resolution, colorSpace, 5)

t0 = time.time()

# Get a camera image.
# image[6] contains the image data passed as an array of ASCII chars.
naoImage = video_service.getImageRemote(videoClient)

t1 = time.time()

# Time the image transfer.
print("acquisition delay ", t1 - t0)

video_service.unsubscribe(videoClient)

imageWidth = naoImage[0]
imageHeight = naoImage[1]
array = naoImage[6]
image_bytes = bytes(array)

# Create a PIL Image from our pixel array.
im = Image.frombytes("RGB", (imageWidth, imageHeight), image_bytes)

# Save the image.
im.save("camImage.png", "PNG")

im.show()

#recive the behaviour detected from the local machine

# behaviour_detected = "nailbiting"

# if (behaviour_detected == "nailbiting"):
#     tts = app.session.service("ALTextToSpeech")
#     tts.say("Please stop biting your nails")
