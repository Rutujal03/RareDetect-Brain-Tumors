# import streamlit as st
# import numpy as np
# from tensorflow.keras.preprocessing import image

# st.title("🧠 Brain Tumor Detection")

# uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

# if uploaded_file is not None:
#     img = image.load_img(uploaded_file, target_size=(224,224))
#     st.image(img)

#     img_array = image.img_to_array(img)/255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     st.success("✅ Image processed successfully")
#     st.info("🤖 Model will be connected after training")


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 🔥 Load all models
vgg_model = load_model("vgg16_model.h5")
resnet_model = load_model("resnet50_model.h5")
rare_model = load_model("raredetect_model.h5")

st.title("🧠 RareDetect: Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128,128))
    st.image(img, caption="Uploaded Image")

    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 Predictions
    vgg_pred = vgg_model.predict(img_array)[0][0]
    resnet_pred = resnet_model.predict(img_array)[0][0]
    rare_pred = rare_model.predict(img_array)[0][0]

    st.success("✅ Image processed successfully")

    # 🎯 Main result (RareDetect)
    if rare_pred > 0.5:
        st.error("🛑 Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")

    # 📊 Percent conversion
    vgg_percent = vgg_pred * 100
    resnet_percent = resnet_pred * 100
    rare_percent = rare_pred * 100

    # 📊 Graph
    models = ['VGG16', 'ResNet50', 'RareDetect']
    values = [vgg_percent, resnet_percent, rare_percent]
    colors = ['#A7C7E7', '#FFD8A8', '#B9FBC0']

    fig, ax = plt.subplots()
    ax.bar(models, values, color=colors)

    ax.set_ylabel("Tumor Probability (%)")
    ax.set_title("Model Comparison")

    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}%", ha='center')

    st.pyplot(fig)

