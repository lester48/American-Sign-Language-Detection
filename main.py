import infer
import streamlit as st
from PIL import Image
import camera
from gtts import gTTS
import os

def main():
    st.title("ASL Recognization APP")

   
    image=Image.open("./images/R.jpg")
    label = infer.predict(image)

    print("Inference Done!")
    print("Label = ", label)
    mytext = 'Identified Label is {}'.format(label)
  

    language = 'en'

    myobj = gTTS(text=mytext, lang=language, slow=False)

    myobj.save("voice.mp3")
  

    os.system("voice.mp3")


if __name__ == '__main__':
    main()