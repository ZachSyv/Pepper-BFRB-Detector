import qi # pip install qi is needed to import this library
import sys 

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

#recive the behaviour detected from the local machine

behaviour_detected = "nailbiting"

if (behaviour_detected == "nailbiting"):
    tts = app.session.service("ALTextToSpeech")
    tts.say("Please stop biting your nails")
