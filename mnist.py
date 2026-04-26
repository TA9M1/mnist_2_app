import os
# メモリ消費とログを最小限に抑える設定
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)
app.secret_key = "aidemy"

classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

# --- 書き込み権限のある /tmp フォルダを利用 ---
UPLOAD_FOLDER = "/tmp/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# モデル保持用の変数
model = None

def get_model():
    global model
    if model is None:
        # Keras 3.0環境で最新のモデル形式を読み込めるようにする
        import keras
        from keras.models import load_model
        try:
            # 互換性向上のため compile=False を推奨
            model = load_model('./model.keras', compile=False)
        except Exception:
            model = load_model('./model.keras')
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 画像処理
            raw_img = Image.open(filepath).convert("RGBA")
            canvas = Image.new("RGBA", raw_img.size, (255, 255, 255))
            canvas.paste(raw_img, mask=raw_img)
            img = canvas.convert("L")
            img = ImageOps.invert(img)
            img = img.point(lambda x: 0 if x < 128 else 255)
            img = img.resize((image_size, image_size))
            
            # 必要なタイミングでKerasの処理をインポート
            import keras
            from keras.utils import img_to_array
            
            img_array = img_to_array(img)
            img_array = img_array / 255.0  
            data = np.expand_dims(img_array, axis=0)

            current_model = get_model()
            # 予測実行（verbose=0で軽量化）
            result = current_model.predict(data, verbose=0)[0]
            predicted = result.argmax()
            pred_answer = f"これは {classes[predicted]} です"

            # 処理後に一時ファイルを削除
            if os.path.exists(filepath):
                os.remove(filepath)

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)