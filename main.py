from flask import Flask, request, send_file, render_template_string
from flask_cors import CORS
import librosa
import soundfile as sf
import numpy as np
import io

app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Suno Audio Anonymization</title>
    <style>
        body { background: #0f172a; color: #fff; font-family: sans-serif; text-align: center; padding: 40px; }
        .card { background: #1e293b; max-width: 600px; margin: 0 auto; padding: 30px; border-radius: 12px; border: 1px solid #334155; }
        input[type="file"] { margin: 20px 0; }
        button { background: #10b981; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-weight: bold; cursor: pointer; }
        button:hover { background: #059669; }
        select { padding: 8px; margin: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Suno Audio Anonymization</h2>
        <p>Tối ưu hóa âm thanh vượt qua bộ lọc Suno AI</p>
        <form action="/process" method="post" enctype="multipart/form-data">
            <input type="file" name="audio" accept="audio/*" required><br>
            <label>Chế độ: </label>
            <select name="preset">
                <option value="subtle">Subtle (Nhẹ - Giữ giọng)</option>
                <option value="moderate" selected>Moderate (Tiêu chuẩn)</option>
                <option value="aggressive">Aggressive (Mạnh mẽ)</option>
            </select><br><br>
            <button type="submit">Bắt đầu xử lý (Anonymize)</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process', methods=['POST'])
def process_audio():
    file = request.files['audio']
    preset = request.form.get('preset', 'moderate')
    
    y, sr = librosa.load(file, sr=44100)
    
    if preset == 'subtle':
        n_steps = 0.4
        rate = 1.01
    elif preset == 'aggressive':
        n_steps = 1.2
        rate = 1.03
    else:
        n_steps = 0.8
        rate = 1.02

    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    y_stretched = librosa.effects.time_stretch(y_shifted, rate=rate)
    
    noise = np.random.randn(len(y_stretched)) * 0.0005
    y_final = y_stretched + noise

    out_io = io.BytesIO()
    sf.write(out_io, y_final, sr, format='WAV', subtype='PCM_16')
    out_io.seek(0)
    
    return send_file(out_io, mimetype="audio/wav", as_attachment=True, download_name="suno_anonymized.wav")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
