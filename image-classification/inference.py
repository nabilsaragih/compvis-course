import os
import joblib
import numpy as np
from PIL import Image
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
import io

class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, target_size=(128, 128), remove_bg=False, temp_dir=None):
        self.target_size = target_size
        self.remove_bg = remove_bg
        self.temp_dir = temp_dir if temp_dir else 'temp_processed_images'
        
        if self.remove_bg and not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        
        for i, image_path in enumerate(X):
            if self.remove_bg:
                img = remove_background(image_path)
                img = img.resize(self.target_size)
                
                feature_vector = self._extract_features_from_image(img)
            else:
                img = resize_image(image_path, self.target_size)
                feature_vector = extract_color_moments(image_path)
                
            features.append(feature_vector)
            
        return np.array(features)
    
    def _extract_features_from_image(self, image):
        image_array = np.array(image)
        
        if len(image_array.shape) == 2: 
            image_array = np.stack((image_array,) * 3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3] 
        
        mean = np.mean(image_array, axis=(0, 1))
        std_dev = np.std(image_array, axis=(0, 1))
        skewness = skew(image_array.reshape(-1, image_array.shape[2]), axis=0)
        
        return np.concatenate([mean, std_dev, skewness])

def resize_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    return image

def remove_background(image_path):
    with open(image_path, 'rb') as input_file:
        input_data = input_file.read()
        from rembg import remove
        output_data = remove(input_data)  
    image = Image.open(io.BytesIO(output_data)).convert("RGBA")
    return image

def extract_color_moments(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    
    if len(image_array.shape) == 2: 
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3] 
    
    mean = np.mean(image_array, axis=(0, 1))
    std_dev = np.std(image_array, axis=(0, 1))
    skewness = skew(image_array.reshape(-1, image_array.shape[2]), axis=0)
    
    return np.concatenate([mean, std_dev, skewness])

def load_butterfly_model(model_path="butterfly_classifier"):
    pipeline = joblib.load(f"model/{model_path}.joblib")
    
    label_map = {}
    with open(f"model/labelmap/{model_path}_labels.txt", 'r') as f:
        for line in f:
            if line.strip():
                label_id, label_name = line.strip().split(':', 1)
                label_map[int(label_id)] = label_name
    
    return pipeline, label_map

def predict_butterfly_class(image_path, model_path="butterfly_classifier"):
    pipeline, label_map = load_butterfly_model(model_path)

    knn_classifier = pipeline.named_steps['classifier']
    features = pipeline.named_steps['feature_extraction'].transform([image_path])
    
    prediction = knn_classifier.predict(features)[0]
    class_name = label_map[prediction]
    neighbors = knn_classifier.kneighbors(features, return_distance=False)[0]
    y_train = knn_classifier._y
    neighbor_votes = [y_train[neighbor] for neighbor in neighbors]
    confidence = neighbor_votes.count(prediction) / len(neighbor_votes)
    
    return class_name, confidence, knn_classifier

if __name__ == "__main__":
    test_image = "data/test/Image_165.jpg"

    class_name, confidence, knn_classifier = predict_butterfly_class(test_image)
    print(f"Predicted butterfly class: {class_name}, confidence: {confidence:.2f}")
    print(f"(KNN vote ratio: {int(confidence * knn_classifier.n_neighbors)}/{knn_classifier.n_neighbors})")