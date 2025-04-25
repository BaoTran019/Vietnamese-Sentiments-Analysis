import os

class BaseModel:
    def __init__(self, model=None):
        self.model = model

    def train(self, X_train, y_train):
        """Training model"""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict with pre-trained model"""
        return self.model.predict(X)

    def save(self, path=None):
            """Lưu mô hình vào thư mục mặc định models/saved_models hoặc theo đường dẫn chỉ định"""
            # Nếu không có path, sử dụng đường dẫn mặc định
            if path is None:
                path = "models/saved_models/my_model.joblib"
            
            # Đảm bảo thư mục tồn tại, nếu không thì tạo mới
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Lưu mô hình
            joblib.dump(self.model, path)
            print(f"Mô hình đã được lưu tại: {path}")

    def load(self, path):
        """Load model from file"""
        import joblib
        self.model = joblib.load(path)
