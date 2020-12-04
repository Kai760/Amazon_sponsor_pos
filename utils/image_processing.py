import cv2


def read_reshape_img(img_path, rate=0.7, color=0):
    # color=1ならカラー、0ならグレースケール
    img = cv2.imread(img_path, color)
    img = cv2.resize(img, (int(img.shape[1] * rate), int(img.shape[0] * rate)))
    return img


def re_reshape_target_img(base_width, target_img, allowed_max_rate=0.25):
    allowed_size = base_width * allowed_max_rate
    target_h, target_w = target_img.shape[:2]
    reduction_rate = max(allowed_size / target_h, allowed_size / target_w)
    if reduction_rate < 1.0:
        target_img = cv2.resize(target_img, (int(target_w * reduction_rate), int(target_h * reduction_rate)))
    return target_img
