import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 余計なログを消す
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # ワンクッション置いてメモリ消費を抑える
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps  # ここに追加
import numpy as np

classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# フォルダがない場合に作成する
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.secret_key = "aidemy" # flashメッセージ表示に必要

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.keras')
model.make_predict_function() # ロード直後に追加
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # --- 画像前処理の開始 ---
            # 1. 透過背景対策：白い背景のキャンバスを作成して貼り付け
            raw_img = Image.open(filepath).convert("RGBA")
            canvas = Image.new("RGBA", raw_img.size, (255, 255, 255))
            canvas.paste(raw_img, mask=raw_img)
            
            # 2. グレースケールに変換し、色を反転（白背景・黒文字 → 黒背景・白文字へ）
            img = canvas.convert("L")
            img = ImageOps.invert(img)

# 文字が細い場合に備えて、少しだけ太くする処理（任意）
            img = img.point(lambda x: 0 if x < 128 else 255)
            # 3. リサイズと正規化
            img = img.resize((image_size, image_size))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  
            data = np.expand_dims(img_array, axis=0)

            # 4. 予測
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")

if __name__ == "__main__":
    # Render等の環境では PORT 環境変数を参照するため
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)