import cv2


class Images(object):
    def __init__(self, whole_img_path, target_img_path, sponsor_img_path, rate):
        self.rate = rate
        self.img_rgb = self._read_reshape_img(whole_img_path, self.rate, color=1)
        self.whole_w = self.img_rgb.shape[1]
        self.whole_img = self._read_reshape_img(whole_img_path, self.rate)
        self.target_img = self._read_reshape_img(target_img_path, self.rate)
        self.sponsor_img = self._read_reshape_img(sponsor_img_path, self.rate)
        self.allowed_size = int(self.whole_w * 0.25)  # これ以上大きい矩形は存在しない説
        self.target_img = self._re_reshape_target_img(self.allowed_size, self.target_img)

    # 画像を読み込んで適宜小さくする
    def _read_reshape_img(self, img_path, rate=0.7, color=0):
        # color=1ならカラー、0ならグレースケール
        img = cv2.imread(img_path, color)
        img = cv2.resize(img, (int(img.shape[1] * rate), int(img.shape[0] * rate)))
        return img

    # 読み込んだtarget画像が大きすぎた場合小さくする
    def _re_reshape_target_img(self, allowed_size, target_img):
        target_h, target_w = target_img.shape[:2]
        reduction_rate = max(allowed_size / target_h, allowed_size / target_w)
        if reduction_rate < 1.0:
            target_img = cv2.resize(target_img, (int(target_w * reduction_rate), int(target_h * reduction_rate)))
        print(target_img.shape)
        return target_img
