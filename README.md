# Face2Voice
#### <i>A web application that synthesizes and outputs voice when you enter face image and text.</i>

We propose a model for synthesizing human voice audio from human face input images and develop a web application that can serve it.<br>
Our model takes human face images and text as inputs. Note that our goal is not to synthesize accurate human voices, but to implement characteristic voice functions that correlate with input images. 

| Input Text | Input Image | Synthesized Voice | Original Voice | 
| :--------- | :---------: | :---------------: | :------------: |
| Put the face image you want to convert to voice and listen to the voice in the text you want. | <img width="100%" src="https://user-images.githubusercontent.com/82092205/212536363-5db6626d-3ad9-4db3-88a8-776d66568d78.png"> | <video src="https://user-images.githubusercontent.com/82092205/212561377-2816cd94-38d4-4bb3-8a57-857281582abe.mp4"/> | <video src="https://user-images.githubusercontent.com/82092205/212561385-32f3b0f6-b19a-4be7-b877-50196fc94afc.mp4"/> |  
| Put the face image you want to convert to voice and listen to the voice in the text you want. | <img width="100%" src="https://user-images.githubusercontent.com/82092205/212536375-b302d319-971d-4630-be76-5e829eb97688.png"> | <video src="https://user-images.githubusercontent.com/82092205/212561410-045ed735-71b9-4212-86fc-2a456f78edab.mp4"/> | <video src="https://user-images.githubusercontent.com/82092205/212561419-909ced93-34c9-4b5a-bb90-203fd06ae94c.mp4"/> |   
| Put the face image you want to convert to voice and listen to the voice in the text you want. | <img width="100%" src="https://user-images.githubusercontent.com/82092205/212536371-a1f630e6-6429-4f48-9602-ef2e2933b663.png"> | <video src="https://user-images.githubusercontent.com/82092205/212561465-1a6076af-367b-4a76-9b43-30d73437ffcd.mp4"> | <video src="https://user-images.githubusercontent.com/82092205/212561454-17e045f9-c727-44cc-974a-91ce567cda8d.mp4">|   
 
 
<b> Video demonstration </b>
<video src="https://user-images.githubusercontent.com/82092205/212548287-20f75bf6-4030-417c-9f89-5c899df5f94e.mp4"> </video> 

Face2Voice project is maintained By [Hyunghee Park](https://github.com/jh5-6) , [Gaeun Kim](https://github.com/nsense-gekim) , [Minha Bae](https://github.com/shin1038)
<br><br>

Face2Voice Model 
--------
Face2Voice is a face speech conversion program, mainly based on [SV2TTS](https://github.com/CorentinJ/Real-Time-Voice-Cloning), [Lip2Speech](https://github.com/Chris10M/Lip2Speech)

#### Face2Voice 
The overall framework is the same as SV2TTS.
Unlike SV2TTS, we use the Speaker Encoder of Lip2Speech because we need to extract the Speaker Vector using the pictures we received as input. <br><br>
<img width="100%" src="https://user-images.githubusercontent.com/82092205/212501409-c95bf11f-6b03-4301-a1a1-a5554e90714c.png">

-----

#### SV2TTS 
[Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis, NIPS 2018.](https://arxiv.org/pdf/1806.04558.pdf)<br>
Learn Speaker verification embeding to create a natural voice with only short voice data <br><br>
<img width="75%" src="https://user-images.githubusercontent.com/82092205/212501466-938ad190-2279-407a-9d81-2d0836924564.png"><br>
 ```
Input : Short speech segments and text
Output : Voice reading (input) text in a voice similar to the input voice
 ```

#### Lip2Speech 
[Show Me Your Face, And I'll Tell You How You Speak, CVPR 2022.](https://arxiv.org/abs/2206.14009)<br>
The identity of the speaker's voice is captured through facial features such as age, gender, and ethnicity, and the voice is generated by conditioning it with the movement of the lips. <br><br>
<img width="75%" src="https://user-images.githubusercontent.com/82092205/212501476-21ca97a6-8439-4c1c-bd00-0949f8101f34.png"><br>
 ```
Input : Mute video
Output : Voice that matches the video
 ```

DataSet
------
#### SV2TTS Model Training with [Librispeech](http://www.openslr.org/12)
<div align="left">
  <img src="https://ifh.cc/g/J6Cbw8.png" width="480"/>
</div>

Librispeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.<br><br>


#### Lip2Speech Model Training with [AVSpeech Dataset](https://looking-to-listen.github.io/avspeech/download.html)  
<div align="left">
  <img src="https://ifh.cc/g/Okw4yo.jpg" width="480"/>
</div>
AVSpeech Dataset is a large collection of video clips of single speakers talking with no audio background interference. The dataset is based on public instructional YouTube videos (talks, lectures, HOW-TOs), from which we automatically extracted short, 3-10 second clips, where the only visible face in the video and audible sound in the soundtrack belong to a single speaking person. Below is a "small" sample of 10,000 clips from the dataset. If you click on a video you will see just a segment from the video on YouTube that is included in the dataset.
 
<br>

Setup 
------
#### 1. Install Requirements 

  1. I recommend setting up a virtual environment using venv, but this is optional.
  2. ```pip install flask```. This is necessary.
  3. ```pip install -r requirements.txt ```

#### 2. Download Pretrained Models
  - [Encoder](https://www.mediafire.com/file/evktjxytts2t72c/lip2speech_final.pth/file) - Copyright (c) 2021 [Christen M](https://github.com/Chris10M)
  - [Synthesizer](https://drive.google.com/file/d/1EqFMIbvxffxtjiVrtykroF6_mUh-5Z3s/view) - Copyright (c) 2019 [Corentin Jemine](https://github.com/CorentinJ)
  - [Vocoder](https://drive.google.com/file/d/1cf2NO6FtI0jDuy8AV3Xgn6leO6dHjIgu/view) - Copyright (c) 2019 [Corentin Jemine](https://github.com/CorentinJ)

Please ensure the files are extracted to these locations within your local copy of the repository:
```
savedmodels/lip2speech_final.pth
savedmodels/synthesizer.pt
savedmodels/vocoder.pt
```

#### 3. Run the application

Use the flask command
```python app.py```
or
```flask run```
<br>
