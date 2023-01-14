# Face2Voice
We propose a model that synthesizes human voice audio from human face input images.<br>
Our model takes human face images and text as inputs. Note that our goal is not to synthesize accurate human voices, but to implement characteristic voice functions that correlate with input images. 

<div align = "left">
  <img src="https://ifh.cc/g/Q72qAF.jpg" width = "480" >
</div>
<br>

<div align="center">
  
  |Image Input|Text Input|Voice Output|
  |---|---|---|
  |<img src="https://ifh.cc/g/f0HMGb.jpg" width="240"/>|Hi, The weather is nice today.|<img src="https://ifh.cc/g/yCo0x3.gif" width="240"/>|  
</div>

Face2Voice project is maintained By [Hyunghee Park](https://github.com/jh5-6) , [Gaeun Kim](https://github.com/nsense-gekim) , [Minha Bae](https://github.com/shin1038)
<br><br>

Face2Voice Model 
--------
Face2Voice is a face speech conversion program, mainly based on [SV2TTS](https://github.com/CorentinJ/Real-Time-Voice-Cloning), [Lip2Speech](https://github.com/Chris10M/Lip2Speech)
<br><br>

DataSet
------
#### SV2TTS 
<div align="left">
  <img src="https://ifh.cc/g/J6Cbw8.png" width="480"/>
</div>
[Librispeech](http://www.openslr.org/12) is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.

#### Lip2Speech 
<div align="left">
  <img src="https://ifh.cc/g/Okw4yo.jpg" width="480"/>
</div>
[AVSpeech Dataset](https://looking-to-listen.github.io/avspeech/download.html) is a large collection of video clips of single speakers talking with no audio background interference. The dataset is based on public instructional YouTube videos (talks, lectures, HOW-TOs), from which we automatically extracted short, 3-10 second clips, where the only visible face in the video and audible sound in the soundtrack belong to a single speaking person. Below is a "small" sample of 10,000 clips from the dataset. If you click on a video you will see just a segment from the video on YouTube that is included in the dataset.

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
