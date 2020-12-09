import cv2
import os
import pandas as pd
import configparser


class Images(object):
    def __init__(self, whole_img_path, target_img_path, sponsor_img_path, rate, ini_path):
        self.ini_path = ini_path
        self.rate = rate
        self.whole_rate, self.target_rate, self.sponsor_rate = self._set_rate(rate)
        # パスの指定
        self.whole_img_path = whole_img_path
        self.target_img_path = target_img_path
        self.sponsor_img_path = sponsor_img_path
        # 画像の読み込み(カラー、全体白黒、ターゲット、スポンサー）
        self.img_rgb = self._read_reshape_img(whole_img_path, self.whole_rate, color=1)
        self.whole_w = self.img_rgb.shape[1]
        self.whole_img = self._read_reshape_img(whole_img_path, self.whole_rate)
        self.target_img = self._read_reshape_img(self.target_img_path, self.target_rate)
        self.allowed_size = int(self.whole_w * 0.25)
        self.target_img = self._re_reshape_target_img(self.allowed_size, self.target_img)
        self.sponsor_img = self._read_reshape_img(self.sponsor_img_path, self.sponsor_rate)

    # 画像を読み込んで適宜小さくする
    def _read_reshape_img(self, img_path, rate, color=0):
        # color=1ならカラー、0ならグレースケール
        img = cv2.imread(img_path, color)
        # 一律rate倍したあと、それぞれを個別rate倍することで同じ画質を再現できる。
        # 画質重視なら最初から個別rate倍をする（これはこれで複数回行えば最適化されるはず）
        # img = cv2.resize(img, (int(img.shape[1] * rate), int(img.shape[0] * rate)))
        img = cv2.resize(img, (int(img.shape[1] * self.rate), int(img.shape[0] * self.rate)))
        img = cv2.resize(img, (int(img.shape[1] * (rate/self.rate)), int(img.shape[0] *(rate/self.rate))))
        return img

    # 読み込んだtarget画像が大きすぎた場合小さくする
    def _re_reshape_target_img(self, allowed_size, target_img):
        target_h, target_w = target_img.shape[:2]
        reduction_rate = max(allowed_size / target_h, allowed_size / target_w)
        if reduction_rate < 1.0:
            target_img = cv2.resize(target_img, (int(target_w * reduction_rate), int(target_h * reduction_rate)))
        return target_img

    def _read_ini(self, path):
        if not os.path.exists(path):
            return ''
        else:
            ini = configparser.ConfigParser()
            ini.read(path)
            return ini

    # 画像を個別に拡縮させることでテンプレートマッチングを一発で決めたい
    def _set_rate(self, rate):
        ini = self._read_ini(self.ini_path)
        if type(ini) == str:
            return rate, rate, rate
        else:
            default = ini['DEFAULT']
            return eval(default['whole']), eval(default['target']), eval(default['sponsor'])

    def _write_file(self, config):
        config.write(open(self.ini_path, 'w'))

    def write_ini(self, whole_w, target_w, sponsor_w):
        config = configparser.ConfigParser()
        new_whole_rate = whole_w / (self.whole_w / self.whole_rate)
        new_target_rate = target_w / (self.target_img.shape[1] / self.target_rate)
        new_sponsor_rate = sponsor_w / (self.sponsor_img.shape[1] / self.sponsor_rate)

        if not os.path.exists(self.ini_path):
            config['DEFAULT']['whole'] = str(new_whole_rate)
            config['DEFAULT']['target'] = str(new_target_rate)
            config['DEFAULT']['sponsor'] = str(new_sponsor_rate)
            self._write_file(config)
        else:
            config.read(self.ini_path)
            config['DEFAULT']['whole'] = str(new_whole_rate)
            config['DEFAULT']['target'] = str(new_target_rate)
            config['DEFAULT']['sponsor'] = str(new_sponsor_rate)
            with open(self.ini_path, "w") as configfile:
                config.write(configfile)
