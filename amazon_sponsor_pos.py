import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse
import time
from utils.image_processing import Images
from utils.template_matching import non_max_supression, delete_not_aimed_sponsor, get_template_pos
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
import datetime
import os
from glob import glob


def setup_logger(log_folder, modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_folder)  # fh = file handler
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger


def print_result(sponsor_pos):
    if sponsor_pos == -1:
        print("target画像が見つかりませんでした。")
    elif sponsor_pos == 0:
        print("SEOで表示された画像はありましたが、スポンサープロダクトではありませんでした。")
    else:
        print(f"ターゲットは{sponsor_pos}番目のスポンサープロダクトです。")


def main(whole_img_path, target_img_path, ini_path, sponsor_img_path='sponsor/sponsor.png'):
    # rate倍された(rgb画像、全体画像、ターゲット画像、スポンサー画像）を所持
    rate = 0.5
    images = Images(whole_img_path, target_img_path, sponsor_img_path, rate, ini_path)

    # テンプレートマッチングにてターゲット画像の位置を取得
    print("ターゲット画像位置取得")
    loc_target, target_w, target_h = get_template_pos(images.whole_img, images.target_img, 0.95, 0.03)
    # もし画像が見つからなければloc_targetは空のタプルを返す
    if len(loc_target) != 2:
        return -1
    loc_target = non_max_supression(loc_target)

    # テンプレートマッチングにてスポンサー画像の位置を取得
    print("\nスポンサープロダクト位置取得")
    # 閾値0.7 &許容範囲0.15でいいのか？？
    loc_sponsor, sponsor_w, sponsor_h = get_template_pos(images.whole_img, images.sponsor_img, 0.7, 0.15)
    loc_sponsor = non_max_supression(loc_sponsor)
    loc_sponsor = delete_not_aimed_sponsor(loc_sponsor, images.whole_w)

    # 取得画像がスポンサープロダクトの何番目かに該当するかを計算
    distance = 10000
    sponsor_pos = 100
    for target_cand in zip(*loc_target[::-1]):
        pos_w, pos_h = target_cand
        cv2.rectangle(images.img_rgb, target_cand, (pos_w + target_w, pos_h + target_h), (0, 0, 255), 2)
        for index, pt in enumerate(zip(*loc_sponsor[::-1])):
            cv2.rectangle(images.img_rgb, pt, (pt[0] + sponsor_w, pt[1] + sponsor_h), (0, 0, 255), 2)
            temp_distance = abs(pt[0] - pos_w) + abs(pt[1] - (pos_h + target_h))
            if temp_distance < distance:
                distance = temp_distance
                sponsor_pos = index + 1
    # スポンサーに対する矩形かどうかを距離から確認
    sponsor_pos = sponsor_pos if images.whole_w * 0.1 > distance else 0

    # 今回の拡縮情報をiniファイルに書き込み
    images.write_ini(images.whole_w, target_w, sponsor_w)

    # 結果確認(デバッグ用)
    cv2.imwrite(f'result_no{sponsor_pos}.png', images.img_rgb)
    return sponsor_pos


def debug(logger):
    products = glob("./dataset/*")
    times = []
    for product in products:
        basename = os.path.basename(product)
        whole_img = f'{product}/whole_{basename}.png'
        target_img = f'{product}/target_{basename}.png'
        ini_path = f'{product}/config.ini'
        start = time.time()

        sponsor_pos = main(whole_img, target_img, ini_path)

        logger.debug(f"{basename}:pos={sponsor_pos}, {time.time() - start}s")
        times.append(time.time() - start)
        print_result(sponsor_pos)
    logger.debug(f"mean = {sum(times) / len(times)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole_img', type=str, default='./dataset/golf/whole_golf.png')
    parser.add_argument('--target_img', type=str, default='./dataset/golf/target_golf.png')
    parser.add_argument('--ini_path', type=str, default='./config.ini')
    opt = parser.parse_args()
    logger = setup_logger(f"{datetime.date.today()}.log")

    # スポンサー順位を取得
    sponsor_pos = main(opt.whole_img, opt.target_img, opt.ini_path)
    logger.debug(f"{os.path.basename(opt.target_img)}:pos={sponsor_pos}")
    print_result(sponsor_pos)

    # debug(logger)
