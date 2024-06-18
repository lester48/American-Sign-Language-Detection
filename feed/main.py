import infer
from PIL import Image
import camera
from gtts import gTTS
import os

def main():
    while True:
        camera.main()
        image=Image.open("./images/image.jpg")
        label = infer.predict(image)
        print("Inference Done!")
        print("Label = ", label)
        mytext = 'Identified Label is {}'.format(label)
        #mytext=label
        language = 'en'
        myobj = gTTS(text=mytext, lang=language, slow=False)
        myobj.save("voice.mp3")
        os.system("voice.mp3")
       


if __name__ == '__main__':
    main()