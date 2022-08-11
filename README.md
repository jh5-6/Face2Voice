# Face2Voice

Setup 
------
### 1. Install Requirements 

  1. I recommend setting up a virtual environment using venv, but this is optional.
  2. ```pip install flask```. This is necessary.
  3. ```pip install -r requirements.txt ```

### 2. Download Pretrained Models
  - [Encoder](https://www.mediafire.com/file/evktjxytts2t72c/lip2speech_final.pth/file) - Copyright (c) 2021 [Christen M](https://github.com/Chris10M)
  - [Synthesizer](https://drive.google.com/file/d/1EqFMIbvxffxtjiVrtykroF6_mUh-5Z3s/view) - Copyright (c) 2019 [Corentin Jemine](https://github.com/CorentinJ)
  - [Vocoder](https://drive.google.com/file/d/1cf2NO6FtI0jDuy8AV3Xgn6leO6dHjIgu/view) - Copyright (c) 2019 [Corentin Jemine](https://github.com/CorentinJ)

Please ensure the files are extracted to these locations within your local copy of the repository:
```
savedmodels/lip2speech_final.pth
savedmodels/synthesizer.pt
savedmodels/vocoder.pt
```

### 3. Run the application

Use the flask command
```python app.py```
or
```flask run```
