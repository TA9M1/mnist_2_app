import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# クラス定義（0〜9の数字）
classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

# アップロード先の指定（Render等の環境でも動くように絶対パスを意識）
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
# flashメッセージ（エラー表示）を使うためにシークレットキーを設定
app.secret_key = "aidemy_mnist_app" 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 学習済みモデルをロード（ファイル名が model.keras であることを確認してください）
model = load_model('./model.keras')

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
            # uploadsフォルダがない場合に備えて作成
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 画像を読み込み、MNISTの学習形式（28x28, グレースケール）に変換
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size, image_size))
            img = image.img_to_array(img)
            # モデルが受け取れる形状 (1, 28, 28, 1) に変換
            data = np.expand_dims(img, axis=0)
            
            # 予測の実行
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは 「" + classes[predicted] + "」 です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")

if __name__ == "__main__":
    # Render等の外部サーバーで動かす際は、ホストとポートを指定するのが一般的
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)