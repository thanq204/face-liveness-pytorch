"""Local Streamlit demo for models trained with personal_liveness.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import streamlit as st
import torch

from personal_liveness import load_classifier, score_video


@st.cache_resource
def cached_classifier(path: Path, modified_ns: int):
    return load_classifier(path)


st.set_page_config(page_title="Face Liveness | Video Demo", layout="centered")
st.title("Face Liveness")
st.caption("Chấm một video bằng checkpoint PyTorch trên máy. Lấy mẫu 2 frame/giây trong 12 giây đầu.")

checkpoints = sorted(Path("artifacts/personal").glob("liveness_*.pt"))
if not checkpoints:
    st.info("Hãy chạy `python personal_liveness.py train` để tạo checkpoint.")
    st.stop()

checkpoint = st.selectbox("Checkpoint", checkpoints, format_func=lambda path: path.stem)
clip = st.file_uploader("Video cần chấm", type=["mp4", "mov", "avi", "mkv"])
if st.button("Phân tích video", disabled=clip is None):
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    with st.spinner("Đang đọc video và chấm điểm…"):
        model, mode = cached_classifier(checkpoint, checkpoint.stat().st_mtime_ns)
        with TemporaryDirectory(prefix="liveness-") as directory:
            path = Path(directory) / ("clip" + Path(clip.name).suffix.lower())
            path.write_bytes(clip.getbuffer())
            result = score_video(path, model, mode)
    if result["label"] is None:
        st.warning("Không có frame đủ điều kiện chấm. Mỗi frame cần đúng một khuôn mặt.")
    else:
        label = "Thật (real)" if result["label"] == "real" else "Giả (fake)"
        left, right = st.columns(2)
        left.metric("Phân loại", label)
        right.metric("Điểm lớp real", f"{result['p_real']:.2f}")
    st.caption(f"Đã chấm {result['scored_frames']}/{result['sampled_frames']} frame lấy mẫu.")
st.caption("Bản thử nghiệm trên dữ liệu tự thu thập, chưa đánh giá để dùng cho quyết định xác thực thực tế.")
