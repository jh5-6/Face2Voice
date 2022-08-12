from fileinput import filename
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

import os 
import numpy as np 
import torch 
import cv2 
import soundfile as sf

from facenet_pytorch import MTCNN
from lip2speech import model as speaker_encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Define a flask app
app = Flask(__name__)
app.secret_key = 'super secret key'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 입력 사진 첨부 시 확장자 제한
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# 입력 사진 저장 폴더
app.config['UPLOAD_FOLDER'] = 'static/images'
# 합성된 음성 저장 폴더
app.config['GENAUDIO_FOLDER'] = 'static/genAudio'

# 입력 이미지 전처리 
# 입력으로 들어온 사진 중 사람의 얼굴 부분만 자르기 위해 사용 
# face detection model
mtcnn = MTCNN( image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

# Face2Voice 구현 시 사용한 3가지 모델 
# Encoder : Lip2Speech의 SpeakerEncoder
net = speaker_encoder.get_network('test')
state_dict = torch.load('savedmodels/lip2speech_final.pth', map_location=device)
if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
speaker_encoding_State_dict = dict()
for k in list(state_dict.keys()):
	if k.startswith('speaker_encoder.'):
		speaker_encoding_State_dict[k[len('speaker_encoder.'):]] = state_dict.pop(k)
net.load_state_dict(state_dict, strict=True)
net.eval()

# Synthesizer : SV2TTS의 synthesizer
synthesizer = Synthesizer('savedmodels/synthesizer.pt')
synthesizer.load()

# Vocoder : SV2TTS의 synthesizer
vocoder.load_model('savedmodels/vocoder.pt')

# 이미지 전처리 
def preprocess_img(fpath):

    img = cv2.imread(fpath)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(img)
    # img = img.astype(np.float64)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], "UseMtcnnToCutOnlyTheFace.jpg")
    mtcnn(img, save_path = image_path, return_prob=True)    
    img_face = cv2.imread(image_path)
    img_face = torch.tensor(img_face).permute(2, 0, 1)
    
    img_aligned = img_face.float() / 255.0 
    aligned = img_aligned.unsqueeze(0)

    return aligned

# 음성 합성
def inference(face_image, text):

    # Speaker Encoding
    speaker_embedding = net.vgg_face.inference(face_image)[0]

    # Sythesize
    texts = text.split("\r\n")
    embeds = [speaker_embedding] * len(texts)

    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)

    # Vocoding 
    wav = vocoder.infer_waveform(spec)

    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    # Trim excessive silences
    # wav = encoder.preprocess_wav(wav)

    wav = wav / np.abs(wav).max() * 0.97
    return wav

# 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def info():
    return render_template('info.html')

@app.route('/main')
def main():

    vocoder._model.progressed_task = 0
    vocoder._model.total_task = 0
    return render_template('main.html')

@app.route('/references')
def references():
    return render_template('references.html')    

@app.route('/progress', methods=["POST"])
def progress():
    return jsonify({
        'progressed' : vocoder._model.progressed_task,
        'total' : vocoder._model.total_task,
    })

 #speech synthesize
@app.route('/result', methods=['POST'])
def result( ):
    if 'imagefile' not in request.files:
        flash('No file part')
        return redirect(request.url)

    imagefile = request.files['imagefile']
    input_text = request.form['inputtext']
    input_text = str(input_text)

    # 입력 사진 선택하지 않은 경우 
    if imagefile.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    # 입력 문장 작성하지 않은 경우 
    if input_text == '':
        flash('No input text written for uploading')
        return redirect(request.url)


    if imagefile and allowed_file(imagefile.filename):
        # 확장자가 png or jpg or jpeg인 경우
        # 입력 사진 저장 
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)
        # flash("Image successfully uploaded and displayed below")

        # 음성 합성 결과 출력 시 
        # 입력으로 넣은 문장 보여주기 위해 result.html에 입력 문장 넘겨줌
        flash(input_text)

        # 전처리한 이미지와 입력 문장을 이용해 음성 합성
        input_img = preprocess_img(image_path)
        wav = inference(input_img, input_text)

        # 합성된 음성 저장 
        imagename = imagefile.filename.split('.')[0]
        genfilename = "genAudio_" + imagename + ".wav"
        audio_path = os.path.join(app.config['GENAUDIO_FOLDER'], genfilename)
        sf.write(audio_path, wav, Synthesizer.sample_rate)
     
        return render_template('result.html', filename = filename, audiofile = genfilename)

    else:
        # flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='images/' + filename), code=301)


@app.route('/play/<audiofile>')
def play(audiofile):
    return redirect(url_for('static', filename='genAudio/' + audiofile), code=301)


if __name__ == "__main__":
    app.debug = True
    app.run()
