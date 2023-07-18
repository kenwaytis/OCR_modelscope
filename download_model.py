from modelscope.hub.snapshot_download import snapshot_download

ocr_detect_dir = snapshot_download("damo/cv_resnet18_ocr-detection-line-level_damo")
ocr_recogenize_dir = snapshot_download("damo/cv_convnextTiny_ocr-recognition-general_damo")