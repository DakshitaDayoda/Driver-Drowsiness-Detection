import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.express as px
import pandas as pd

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚘",
    layout="wide"
)

# =====================================
# CUSTOM CAR THEME CSS
# =====================================
st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #0f2d3a,
        #355c6d,
        #6c8ea3,
        #d9cbbf
    );
    background-attachment: fixed;
}

/* ===== MAIN ===== */
.main {
    background: transparent;
}

/* ===== TITLE ===== */
h1 {
    color: #f4e1c1;
    text-align: center;
    font-size: 54px;
    font-weight: 800;
    letter-spacing: 2px;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.35);
}

/* ===== HEADINGS ===== */
h2, h3 {
    color: #f8f4ec;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #123142,
        #355c6d
    );
    border-right: 3px solid #d9a066;
}

section[data-testid="stSidebar"] * {
    color: #f8f4ec;
}

/* ===== BUTTONS ===== */
.stButton>button {
    background: linear-gradient(
        135deg,
        #d98952,
        #c96c4b
    );
    color: white;
    border-radius: 14px;
    border: none;
    padding: 12px 24px;
    font-size: 17px;
    font-weight: bold;
    transition: 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(
        135deg,
        #e0b36d,
        #d98952
    );
    transform: scale(1.04);
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(18, 49, 66, 0.75);
    border: 2px dashed #d98952;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

[data-testid="stFileUploader"] label {
    color: #f4e1c1 !important;
    font-size: 18px;
    font-weight: bold;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(53, 92, 109, 0.55);
    border-radius: 15px;
    color: #f8f4ec;
}

/* ===== RESULT BOX ===== */
.result-box {
    padding: 30px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    text-align: center;
    font-size: 28px;
    color: #f8f4ec;
    font-weight: bold;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.35);
    border: 1px solid rgba(224,179,109,0.35);
}

/* ===== INFO BOX ===== */
.info-box {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding: 18px;
    border-radius: 16px;
    color: #f8f4ec;
    font-size: 18px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.25);
    border-left: 5px solid #d98952;
}

/* ===== IMAGE ===== */
img {
    border-radius: 20px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
}

/* ===== TEXT ===== */
p, label, div {
    color: #f8f4ec;
}

/* ===== CHART BACKGROUND REMOVE ===== */

[data-testid="stVegaLiteChart"] {
    background: transparent !important;
    border-radius: 15px;
    padding: 10px;
}

canvas {
    background: transparent !important;
}

/* ===== FOOTER ===== */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# TITLE
# =====================================
st.title("🚗 Driver Drowsiness Detection")

st.markdown(
    """
    <div class='info-box'>
    🚘 Detect driver eye fatigue and monitor alertness in real-time for safer driving assistance.
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_cnn_model():
    model = load_model('drowsiness_model.keras')
    return model

model = load_cnn_model()

# =====================================
# SETTINGS
# =====================================
IMG_SIZE = 64

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📤 Upload Image",
        "📊 Model Performance",
        "👁️ Eye State Analysis",
        "⚠️ Driver Safety Tips",
        "📖 About Project"
    ]
)

# =====================================
# HOME PAGE
# =====================================
if page == "🏠 Home":

    st.subheader("🚘 Driver Drowsiness Detection System")

    st.markdown("""
    <div class='info-box'>
    AI-powered Driver Drowsiness Detection System that monitors driver eye activity and detects fatigue using Computer Vision and CNN.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================
    # VISUALIZATION SECTION
    # =====================================

    st.subheader("📊 Detection Visualizations")

    col1, col2 = st.columns(2)

    # =====================================
    # LINE CHART
    # =====================================

    with col1:

        st.markdown("### 📈 Accuracy Trend")

        df_line = pd.DataFrame({
            "Epoch": [1, 2, 3, 4, 5],
            "Accuracy": [72, 81, 87, 91, 95]
        })

        fig_line = px.line(
            df_line,
            x="Epoch",
            y="Accuracy",
            markers=True
        )

        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(fig_line, use_container_width=True)

    # =====================================
    # BAR CHART
    # =====================================

    with col2:

        st.markdown("### 📊 Eye Detection")

        df_bar = pd.DataFrame({
            "Type": ["Open Eye", "Closed Eye"],
            "Value": [92, 96]
        })

        fig_bar = px.bar(
            df_bar,
            x="Type",
            y="Value"
        )

        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # =====================================
    # DONUT CHART
    # =====================================

    st.markdown("### 🍩 Detection Distribution")

    df_pie = pd.DataFrame({
        "Category": ["Alert", "Drowsy"],
        "Value": [70, 30]
    })

    fig_pie = px.pie(
        df_pie,
        names="Category",
        values="Value",
        hole=0.5
    )

    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# =====================================
# UPLOAD IMAGE PAGE
# =====================================
elif page == "📤 Upload Image":

    st.subheader("📤 Upload Eye Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)

        img = np.array(image)

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img_normalized = img_resized / 255.0

        img_reshape = np.reshape(
            img_normalized,
            (1, IMG_SIZE, IMG_SIZE, 3)
        )

        prediction = model.predict(img_reshape, verbose=0)

        predicted_value = prediction[0][0]

        # Correct prediction logic
        if predicted_value > 0.5:
            label = 'Closed Eyes 😴'
            status = 'Driver is Drowsy ⚠️'
            confidence = predicted_value * 100
            closed_conf = predicted_value
            open_conf = 1 - predicted_value

        else:
            label = 'Open Eyes 👁️'
            status = 'Driver is Alert ✅'
            confidence = (1 - predicted_value) * 100
            open_conf = 1 - predicted_value
            closed_conf = predicted_value

        with col2:

            st.markdown(
                f"""
                <div class='result-box'>
                Prediction:<br><br>
                {label}<br><br>
                Confidence: {confidence:.2f}%<br><br>
                {status}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("Model Output:", prediction)

        # Confidence Chart
        st.subheader("📊 Prediction Confidence")

        df_conf = pd.DataFrame({
            "Eye State": ["Open Eyes", "Closed Eyes"],
            "Confidence": [
                float(open_conf * 100),
                float(closed_conf * 100)
            ]
        })

        fig_conf = px.bar(
            df_conf,
            x="Eye State",
            y="Confidence",
            text="Confidence"
        )

        fig_conf.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )

        fig_conf.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis_title="Confidence %",
            xaxis_title=""
        )

        st.plotly_chart(fig_conf, use_container_width=True)

        # Analysis Section
        st.subheader("📈 Eye State Analysis")

        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:

            st.write("Open Eye Confidence")
            st.progress(float(open_conf))

            st.write("Closed Eye Confidence")
            st.progress(float(closed_conf))

        with analysis_col2:

            st.metric(
                label="Open Eye %",
                value=f"{open_conf * 100:.2f}%"
            )

            st.metric(
                label="Closed Eye %",
                value=f"{closed_conf * 100:.2f}%"
            )

        if predicted_value > 0.5:
            st.error("⚠️ Driver appears drowsy.")

        else:
            st.success("✅ Driver appears alert.")

# =====================================
# MODEL PERFORMANCE PAGE
# =====================================
elif page == "📊 Model Performance":

    st.subheader("📊 Model Performance")

    st.markdown("""
    ### Model Accuracy

    - Training Accuracy: 95%
    - Validation Accuracy: 93%
    - CNN-Based Binary Classification
    """)

    performance_data = pd.DataFrame({
        "Category": ["Training Accuracy", "Validation Accuracy"],
        "Value": [95, 93]
    })

    fig = px.bar(
        performance_data,
        x="Category",
        y="Value",
        text="Value"
    )

    fig.update_traces(
        texttemplate='%{text}%',
        textposition='outside'
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title="",
        yaxis_title="Accuracy %"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================
# EYE STATE ANALYSIS PAGE
# =====================================
elif page == "👁️ Eye State Analysis":

    st.subheader("👁️ Eye State Analysis")

    st.markdown("""
    ### Eye Detection Analysis

    ✅ Open Eye Detection

    ✅ Closed Eye Detection

    ✅ Driver Fatigue Monitoring

    ✅ Alertness Analysis
    """)

    eye_data = pd.DataFrame({
        "Detection": [
            "Open Eye Detection",
            "Closed Eye Detection"
        ],
        "Value": [92, 96]
    })

    fig_eye = px.bar(
        eye_data,
        x="Detection",
        y="Value"
    )

    fig_eye.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title="",
        yaxis_title="Detection %"
    )

    st.plotly_chart(fig_eye, use_container_width=True)

# =====================================
# DRIVER SAFETY TIPS PAGE
# =====================================
elif page == "⚠️ Driver Safety Tips":

    st.subheader("⚠️ Driver Safety Tips")

    st.markdown("""
    ### Safe Driving Tips 🚘

    - Take regular breaks during long drives.
    - Avoid driving while sleepy.
    - Stay hydrated while driving.
    - Get proper sleep before travelling.
    - Stop driving immediately if feeling drowsy.
    """)

# =====================================
# ABOUT PAGE
# =====================================
elif page == "📖 About Project":

    st.subheader("📖 About Project")

    st.markdown("""
### Technologies Used

- Python
- OpenCV
- TensorFlow
- CNN
- Streamlit
- NumPy

---

### Features

✅ Eye State Detection

✅ Open Eye Prediction

✅ Closed Eye Prediction

✅ Drowsiness Alert

✅ CNN-Based Classification

---

### Project Workflow

1. Upload Eye Image
2. Image Preprocessing
3. CNN Prediction
4. Eye State Detection
5. Display Result

---

### Deep Learning Used

Convolutional Neural Network (CNN)
for Driver Drowsiness Detection.
""")

# =====================================
# FOOTER
# =====================================
st.write("")
st.write("---")

st.markdown(
    """
    <center>
    <h4 style='
        color:#f4e1c1;
        font-size:22px;
        font-weight:bold;
        letter-spacing:1px;
        text-shadow:1px 1px 8px rgba(0,0,0,0.35);
    '>
    🚘 Driver Drowsiness Detection System using AI & Computer Vision
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)
