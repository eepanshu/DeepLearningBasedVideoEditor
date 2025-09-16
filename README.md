# Deep Learning Based Video Editor

A cutting-edge video editing tool powered by deep learning. Remove objects, change backgrounds, and process videos with RTX acceleration—all through an intuitive Gradio web interface.

---

## 🚀 Features

- **Object Removal:** Detect and remove objects (e.g., "Remove cup and vase") using YOLOv8 and SAM.
- **Background Replacement:** Switch backgrounds to dark, light, or custom images/videos.
- **GPU Acceleration:** Automatic CUDA support for fast processing on RTX GPUs.
- **Gradio Web UI:** Edit videos easily from your browser—no coding required.
- **Customizable:** Extendable for new object classes and background types.

---

## 🖥️ Gradio Interface

- **Video Upload:** Drag & drop or select a video file.
- **Edit Description:** Type what objects to remove or background changes (e.g., "Remove person and set background to dark").
- **Background Options:** Choose between none, dark, light, or upload a custom background.
- **Live Preview:** See the edited video directly in your browser.

![Gradio UI Screenshot](docs/gradio_ui.png) <!-- Add a screenshot if available -->

---

## 📦 Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/eepanshu/DeepLearningBasedVideoEditor.git
    cd DeepLearningBasedVideoEditor
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download models:**
    - Place `yolov8s.pt` and `sam_vit_b.pth` in the `models/` folder (not tracked by git).

---

## 🏃‍♂️ Usage

1. **Activate your virtual environment (optional):**
    ```sh
    .\venv_ve\Scripts\activate
    ```

2. **Run the app:**
    ```sh
    python app.py
    ```

3. **Open the Gradio UI:**
    - The app will launch a local server and display a link (e.g., `http://localhost:7860`).
    - Open the link in your browser.

---

## 🗂️ Project Structure

```
Video_editor/
├── app.py                # Main application
├── models/               # Pretrained model files (excluded from git)
├── outputs/              # Edited videos (excluded from git)
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes models/ and outputs/
└── README.md             # Project documentation
```

---

## ⚡ Requirements

- Python 3.8+
- NVIDIA GPU (RTX recommended for acceleration)
- [YOLOv8](https://github.com/ultralytics/ultralytics) and [SAM](https://github.com/facebookresearch/segment-anything) model weights

---

## 📝 Customization

- **Add new object classes:** Edit `get_target_objects()` in `app.py`.
- **Change UI layout:** Modify Gradio blocks in `app.py`.
- **Advanced editing:** Integrate more models or effects as needed.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue for major changes.

---

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [Gradio](https://gradio.app/)

---

**Enjoy fast, AI-powered video editing!**
