# # ---------------------------------------------------------------
# # Eagle Vision -- YOLOv8 Training Script for Google Colab (T4 GPU)
# # ---------------------------------------------------------------
# # Run each section as a separate Colab cell.
# # Estimated training time: ~20-30 minutes on T4.

# # -- Cell 1: Verify GPU --------------------------------------
# !nvidia-smi

# # -- Cell 2: Install dependencies ----------------------------
# !pip install ultralytics roboflow
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# import torch
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - change runtime to T4!'}")

# # -- Cell 3: Download dataset from Roboflow ------------------
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
# project = rf.workspace("project-dvd12").project("construction-equipment-b9fth")
# version = project.version(2)
# dataset = version.download("yolov8")
# print(f"Dataset location: {dataset.location}")

# # -- Cell 4: Train -------------------------------------------
# from ultralytics import YOLO

# model = YOLO("yolov8s.pt")

# results = model.train(
#     data=f"{dataset.location}/data.yaml",
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     device=0 if torch.cuda.is_available() else "cpu",
#     workers=2,
#     patience=15,
#     optimizer="AdamW",
#     lr0=0.001,
#     lrf=0.01,
#     weight_decay=0.0005,
#     warmup_epochs=5,
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=10.0,
#     translate=0.1,
#     scale=0.5,
#     fliplr=0.5,
#     mosaic=1.0,
#     mixup=0.1,
#     copy_paste=0.1,
#     project="eagle_vision",
#     name="yolov8s_construction",
#     save=True,
#     save_period=10,
#     plots=True,
#     verbose=True,
# )

# # -- Cell 5: Validate best model ----------------------------
# best_model = YOLO("eagle_vision/yolov8s_construction/weights/best.pt")
# metrics = best_model.val()
# print(f"mAP50:    {metrics.box.map50:.4f}")
# print(f"mAP50-95: {metrics.box.map:.4f}")

# # -- Cell 6: Run inference on a test image -------------------
# import glob
# import matplotlib.pyplot as plt
# from PIL import Image

# test_images = glob.glob(f"{dataset.location}/test/images/*.jpg")[:4]

# fig, axes = plt.subplots(1, len(test_images), figsize=(20, 5))
# if len(test_images) == 1:
#     axes = [axes]

# for ax, img_path in zip(axes, test_images):
#     preds = best_model.predict(source=img_path, conf=0.35, verbose=False)
#     annotated = preds[0].plot()
#     ax.imshow(annotated[:, :, ::-1])
#     ax.axis("off")

# plt.tight_layout()
# plt.show()

# # -- Cell 7: Export best model -------------------------------
# best_model.export(format="onnx")
# print("Best weights: eagle_vision/yolov8s_construction/weights/best.pt")

# # -- Cell 8: Download best.pt to local machine --------------
# from google.colab import files
# files.download("eagle_vision/yolov8s_construction/weights/best.pt")

# # ---------------------------------------------------------------
# # Tips to improve performance later:
# # ---------------------------------------------------------------
# # 1. More data       -- aim for 1500+ annotations per class
# # 2. Hard negatives  -- add frames with NO equipment to reduce FPs
# # 3. Tile inference  -- use SAHI for detecting small/distant equipment
# # 4. Pseudo-labels   -- run best.pt on unlabeled video, review & add
# # 5. Class-specific  -- if one class lags, add more samples for it
# # 6. TTA at val time -- model.val(augment=True) for a free mAP boost
# # ---------------------------------------------------------------
