# ASL-Recognization
In this project, I have applied transfer learning to an Inception-v3 Model to recognize Alphabets and Numbers in American Sign Language with 99% accuracy on the test set. You can find the training notebook [here](https://github.com/AyushiNM/ASL-Recognization/blob/main/Sign_Language_Alphabet_Recognizer.ipynb).

To implement this code, you can -
1. Download the model [here](https://drive.google.com/file/d/1zrpDYeS7AXeGmO4G3FAD55LY53PZZ7fh/view?usp=sharing).<br>

2. I have deployed the app using StreamLit, where you can directly upload a picture and you'll get the results. To check by uploading images run -<br>
`streamlit run main.py`<br>

3. I have also implemened the real-time ASL detection using Google's [mediapipe](https://google.github.io/mediapipe/), which assists in hand detection. To see live ASL recoginition run -<br>
`python live.py` <br>

## References
* https://google.github.io/mediapipe/
* https://streamlit.io/
* https://github.com/loicmarie/sign-language-alphabet-recognizer/archive/refs/heads/master.zip
* https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899/4
* https://www.kaggle.com/leifuer/intro-to-pytorch-loading-image-data
