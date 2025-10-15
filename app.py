import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import numpy as np
import cv2

# 🧠 Load mô hình MobileNetV2
@st.cache_resource
def load_model():
    model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
    state_dict = torch.load("best_mobilenetv2.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# 🏷️ Danh sách lớp bệnh
class_names = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf Blast",
    "Leaf Scald", "Narrow Brown Leaf Spot", "Neck Blast", "Rice Hispa",
    "Sheath Blight", "Tungro Disease"
]

# 📚 Thông tin bệnh
rice_disease_info = {
    "Bacterial Leaf Blight": {
        "vi": "Bệnh bạc lá: Lá bị vàng, có vệt nước, vi khuẩn gây hại.",
        "solution": "Dùng thuốc gốc đồng, kiểm tra nguồn nước."
    },
    "Brown Spot": {
        "vi": "Bệnh đốm nâu: Xuất hiện đốm tròn nâu trên lá.",
        "solution": "Phun thuốc trừ nấm, tăng cường phân kali."
    },
    "Healthy Rice Leaf": {
        "vi": "Lá khỏe mạnh: Không có dấu hiệu bệnh.",
        "solution": "Duy trì chế độ chăm sóc hiện tại."
    },
    "Leaf Blast": {
        "vi": "Bệnh đạo ôn lá: Vết cháy hình thoi, lan nhanh.",
        "solution": "Phun thuốc trừ nấm, kiểm tra độ ẩm ruộng."
    },
    "Leaf Scald": {
        "vi": "Bệnh cháy bìa lá: Mép lá bị cháy, khô.",
        "solution": "Giảm phân đạm, tăng kali, kiểm tra nước."
    },
    "Narrow Brown Leaf Spot": {
        "vi": "Bệnh đốm nâu hẹp: Vết đốm dài, hẹp màu nâu.",
        "solution": "Phun thuốc trừ nấm, cải thiện đất."
    },
    "Neck Blast": {
        "vi": "Bệnh đạo ôn cổ bông: Cổ bông bị thối, không trổ hạt.",
        "solution": "Phun thuốc trừ nấm trước trổ, kiểm tra giống lúa."
    },
    "Rice Hispa": {
        "vi": "Sâu cuốn lá lúa: Lá bị cuốn, có sâu bên trong.",
        "solution": "Phun thuốc trừ sâu, bắt sâu thủ công."
    },
    "Sheath Blight": {
        "vi": "Bệnh khô vằn: Vết bệnh ở bẹ lá, lan lên lá.",
        "solution": "Phun thuốc trừ nấm, giảm mật độ cấy."
    },
    "Tungro Disease": {
        "vi": "Bệnh vàng lùn-lùn xoắn lá: Lá xoắn, cây lùn, virus gây hại.",
        "solution": "Diệt rầy truyền bệnh, dùng giống kháng bệnh."
    }
}

# 🌫️ Làm mờ ảnh bằng Fuzzy Filter
def fuzzy_mean_filter(image: Image.Image, kernel_size=5, sigma=1.2):
    img_np = np.array(image)
    k = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = k @ k.T
    filtered = np.zeros_like(img_np)
    for c in range(3):
        filtered[:, :, c] = cv2.filter2D(img_np[:, :, c], -1, kernel)
    return Image.fromarray(filtered)

# 🔍 Dự đoán và Grad-CAM
def run_gradcam(image: Image.Image):
    try:
        image_filtered = fuzzy_mean_filter(image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image_filtered).unsqueeze(0)

        feature_maps = []
        gradients = []

        def forward_hook(module, input, output):
            feature_maps.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = model.features[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        pred_label = class_names[pred_class] if confidence > 0.5 else "Không xác định"

        model.zero_grad()
        output[0, pred_class].backward()

        grads = gradients[0].mean(dim=[2, 3], keepdim=True)
        fmap = feature_maps[0]
        cam = (grads * fmap).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0 and not torch.isnan(cam.max()):
            cam = cam / cam.max()
        else:
            cam = torch.zeros_like(cam)
        cam = cam.cpu().numpy()

        cam = cv2.resize(cam, image.size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        img_np = np.array(image)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

        return overlay, pred_label
    except Exception:
        return np.array(image), "Không xác định"

# 🎨 Giao diện Streamlit
st.set_page_config(page_title="🌾 Nhận diện bệnh lúa", layout="wide")
st.title("🌾 Nhận diện bệnh lúa bằng MobileNetV2 + Grad-CAM")

uploaded_file = st.file_uploader("📤 Kéo thả ảnh lá lúa tại đây", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Ảnh gốc", use_column_width=True)

    overlay, label = run_gradcam(image)
    st.image(overlay, caption=f"🔥 Grad-CAM: {label}", use_column_width=True)

    info = rice_disease_info.get(label, {
        "vi": "Không có thông tin bệnh.",
        "solution": "Vui lòng kiểm tra lại ảnh hoặc mô hình."
    })

    st.markdown(f"### 🔍 Dự đoán: `{label}`")
    st.markdown(f"**📖 Mô tả:** {info['vi']}")
    st.markdown(f"**🛡️ Cách phòng chống:** {info['solution']}")
