import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse
import time
from utils.image_processing import read_reshape_img, re_reshape_target_img
from utils.template_matching import non_max_supression, delete_not_aimed_sponsor, get_template_pos

def print_result(sponsor_pos, start):
    if sponsor_pos == -1:
        print("target画像が見つかりませんでした。")
        print(f"かかった時間は{round(time.time() - start, 2)}秒です")
    else:
        print(f"\nターゲットは{sponsor_pos}番目のスポンサープロダクトです。")
        print(f"かかった時間は{round(time.time() - start, 2)}秒です")


def main(whole_img_path, target_img_path, sponsor_img='sponsor/sponsor.png'):
    # 後の計算時間を考えてサイズをrate倍にする
    rate = 0.5
    img_rgb = read_reshape_img(whole_img_path, rate, color=1)
    whole_img = read_reshape_img(whole_img_path, rate)
    target_img = read_reshape_img(target_img_path, rate)
    target_img = re_reshape_target_img(whole_img.shape[1], target_img, allowed_max_rate=0.25)
    sponsor_img = read_reshape_img(sponsor_img, rate)

    # ターゲット画像のサイズ感と全体画像のサイズ感を初めからある程度合わせたい。
    # その他高速化手法を要検討
    print("ターゲット画像位置取得")
    loc_target, target_w, target_h = get_template_pos(whole_img, target_img, 0.95, 0.03)
    if len(loc_target) !=2:  # もし画像が見つからなければloc_targetは空のタプルを返す
        return -1
    loc_target = non_max_supression(loc_target)

    print("\nスポンサープロダクト位置取得")
    loc_sponsor, sponsor_w, sponsor_h = get_template_pos(whole_img, sponsor_img, 0.7, 0.15)
    loc_sponsor = non_max_supression(loc_sponsor)
    loc_sponsor = delete_not_aimed_sponsor(loc_sponsor, rate)

    # スポンサープロダクトの何番目かを計算
    distance = 10000
    sponsor_pos = 100
    for target_cand in zip(*loc_target[::-1]):
        pos_w = target_cand[0]
        pos_h = target_cand[1]
        cv2.rectangle(img_rgb, target_cand, (pos_w + target_w, pos_h + target_h), (0, 0, 255), 2)
        for index, pt in enumerate(zip(*loc_sponsor[::-1])):
            cv2.rectangle(img_rgb, pt, (pt[0] + sponsor_w, pt[1] + sponsor_h), (0, 0, 255), 2)
            temp_distance = abs(pt[0] - pos_w) + abs(pt[1] - (pos_h + target_h))
            if temp_distance < distance:
                distance = temp_distance
                sponsor_pos = index + 1

    # 結果を確認
    cv2.imwrite(f'result_no{sponsor_pos}.png', img_rgb)
    return sponsor_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole_img', type=str, default='./data/whole_yoga.png')
    parser.add_argument('--target_img', type=str, default='./data/target_yoga.png')
    opt = parser.parse_args()
    start = time.time()
    sponsor_pos = main(opt.whole_img, opt.target_img)
    print_result(sponsor_pos, start)
