def get_model():
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        # compile=False で学習用データを読み込まず軽量化
        model = load_model('./model.keras', compile=False)
    return model