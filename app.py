import streamlit as st
import joblib
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# --- CONFIGURATION ---
IMAGE_SIZE = (224, 224)
MODEL_PATH = "voting_tri_model_resnet50_classifier.pkl"
SCALER_PATH = "resnet50_feature_scaler.pkl"
CLASS_NAMES = ["แอปเปิ้ล (Apple)", "ส้ม (Orange)", "มะม่วง (Mango)", "กล้วย (Banana)"]

# Define the classification dictionary (4 classes)
class_dict = {i: name for i, name in enumerate(CLASS_NAMES)}

# --- 1. Load Assets (Voting Classifier and Scaler) ---
@st.cache_resource
def load_assets():
    """Loads the model, scaler, and ResNet50 feature extractor."""
    try:
        # Load the final Voting Classifier model
        voting_clf = joblib.load(MODEL_PATH)

        # Load the fitted StandardScaler used during training
        scaler = joblib.load(SCALER_PATH)

        # Initialize ResNet50 Feature Extractor
        # We must use the exact same feature extractor used during training
        base_model = ResNet50(weights='imagenet', include_top=True)
        # Output from the Global Average Pooling layer (layer index -2) -> 2048 features
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

        return voting_clf, scaler, feature_extractor

    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Please ensure both {MODEL_PATH} and {SCALER_PATH} exist after training.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models or ResNet50: {e}")
        st.stop()

voting_clf, scaler, feature_extractor = load_assets()


# --- 2. UI Layout ---
st.title("Fruit Classifier 🍎🍊🥭🍌")
st.markdown("ระบบจำแนกผลไม้ 4 ชนิด (Apple, Orange, Mango, Banana) โดยใช้เทคนิค Transfer Learning (ResNet50 Features + Voting Classifier)")

uploaded_file = st.file_uploader(
    "Upload a fruit image...", 
    type=["jpg", "png", "jpeg"]
)

# --- 3. Prediction Logic ---

if uploaded_file is not None:
    # Load and Display Image
    image = Image.open(uploaded_file).convert("RGB")

    # Display the image before prediction button
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict Fruit"):
        with st.spinner("Processing image and running ensemble prediction..."):
            try:
                # --- A. Preprocess for ResNet50 ---

                # 1. Convert PIL Image to BGR Numpy array (OpenCV/Keras default)
                image_array_rgb = np.array(image)
                image_array_bgr = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)

                # 2. Resize to 224x224 (ResNet50 requirement)
                image_resized = cv2.resize(image_array_bgr, IMAGE_SIZE) 

                # 3. Add batch dimension and run Keras preprocessing
                img_array_batch = np.expand_dims(image_resized, axis=0)
                processed_img = preprocess_input(img_array_batch)

                # --- B. Feature Extraction ---

                # 4. Extract 2048 features using the ResNet50 model
                feature_vector = feature_extractor.predict(processed_img, verbose=0)

                # --- C. Scaling ---

                # 5. Apply the StandardScaler fitted during training
                feature_scaled = scaler.transform(feature_vector)

                # --- D. Predict with Voting Classifier ---

                # 6. Predict the class label
                prediction = voting_clf.predict(feature_scaled)[0]

                # Get the prediction name (e.g., "แอปเปิ้ล")
                prediction_name = class_dict[prediction]

                # Get prediction probabilities for confidence
                probabilities = voting_clf.predict_proba(feature_scaled)[0]
                confidence = np.max(probabilities) * 100

                st.success(f"**ผลการทำนาย (Prediction Result):** **{prediction_name}**")
                st.info(f"ความเชื่อมั่น (Confidence): **{confidence:.2f}%**")

            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                st.text("Check console for technical details.")
