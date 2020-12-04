import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse
import time
from utils.image_processing import Images
from utils.template_matching import non_max_supression, delete_not_aimed_sponsor, get_template_pos


def print_result(sponsor_pos, start):
    if sponsor_pos == -1:
        print("target画像が見つかりませんでした。")
        print(f"かかった時間は{round(time.time() - start, 2)}秒です")
    else:
        print(f"\nターゲットは{sponsor_pos}番目のスポンサープロダクトです。")
        print(f"かかった時間は{round(time.time() - start, 2)}秒です")


def main(whole_img_path, target_img_path, sponsor_img_path='sponsor/sponsor.png'):
    # 0.5倍で大丈夫なのかは要検討。元のスポンサーの縦が20pxlsしかないから半分だと結構潰れる
    rate = 0.5
    # rate倍された(rgb画像、全体画像、ターゲット画像、スポンサー画像）を所持
    images = Images(whole_img_path, target_img_path, sponsor_img_path, rate)

    # その他高速化手法を要検討
    print("ターゲット画像位置取得")
    loc_target, target_w, target_h = get_template_pos(images.whole_img, images.target_img, 0.95, 0.03)
    if len(loc_target) != 2:  # もし画像が見つからなければloc_targetは空のタプルを返す
        return -1
    loc_target = non_max_supression(loc_target)

    print("\nスポンサープロダクト位置取得")
    # 閾値0.7 &許容範囲0.15でいいのか？？
    loc_sponsor, sponsor_w, sponsor_h = get_template_pos(images.whole_img, images.sponsor_img, 0.7, 0.15)
    loc_sponsor = non_max_supression(loc_sponsor)
    loc_sponsor = delete_not_aimed_sponsor(loc_sponsor, images.whole_w)

    # スポンサープロダクトの何番目かを計算
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

    # 結果を確認
    cv2.imwrite(f'result_no{sponsor_pos}.png', images.img_rgb)
    return sponsor_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole_img', type=str, default='./data/whole_display.png')
    parser.add_argument('--target_img', type=str, default='./data/target_display.png')
    opt = parser.parse_args()
    start = time.time()
    sponsor_pos = main(opt.whole_img, opt.target_img)
    print_result(sponsor_pos, start)
