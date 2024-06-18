import infer
import streamlit as st
from PIL import Image
import camera
from gtts import gTTS
import os

def main():
    st.title("ASL Recognization APP")

    uploaded_file = st.file_uploader("Upload an image here -")
    label = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Alphabet.', width=200)
        st.write("")
        label = infer.predict(image)

        st.write("Inference Done!")
        output = "Label = " + label
        st.write(output)
    #camera.main()
    #image=Image.open("./images/L.jpg")
    #label = infer.predict(image)

    print("Inference Done!")
    print("Label = ", label)
    mytext = 'Identified Label is {}'.format(label)
  

    language = 'en'

    myobj = gTTS(text=mytext, lang=language, slow=False)

    myobj.save("voice.mp3")
  

    os.system("voice.mp3")


if __name__ == '__main__':
    main()