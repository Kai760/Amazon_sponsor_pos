import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse


def delete_left_sponsor(loc_sponsor):
    pos = np.argmin(loc_sponsor[1])
    loc_sponsor = np.delete(loc_sponsor, pos, axis=1)
    return loc_sponsor


def read_reshape_img(img_path, rate=0.7, color=0):
    # グレースケールで読み込む
    img = cv2.imread(img_path, color)
    img = cv2.resize(img, (int(img.shape[1] * rate), int(img.shape[0] * rate)))
    return img


def get_template_img_pos(whole_img, target_img, threshold_goal, quality):
    count = 1
    target_org = target_img
    org_w, org_h = target_org.shape[::-1]
    threshold = 0
    while threshold < threshold_goal:
        res = cv2.matchTemplate(whole_img, target_img, cv2.TM_CCOEFF_NORMED)
        threshold = res.max() - quality
        print(f"\r{threshold}", end="")
        coef = count * 0.01
        w, h = target_img.shape[::-1]
        if count % 2 == 0:
            target_img = cv2.resize(target_org, (int(org_w * (1 - coef)),
                                                 int(org_h * (1 - coef))))
        else:
            target_img = cv2.resize(target_org, (int(org_w * (1 + coef)),
                                                 int(org_h * (1 + coef))))
        count += 1
    return threshold, res, w, h


def non_max_supression(loc):
    pts = []
    for index, pt in enumerate(zip(*loc[::])):
        if index == 0:
            pts.append(pt)
        min_dis = min([abs(pt[0] - pt_past[0]) + abs(pt[1] - pt_past[1])
                   for pt_past in pts])
        if min_dis > 30:
            pts.append(pt)
    y = np.array([pt[0] for pt in pts])
    x = np.array([pt[1] for pt in pts])
    return (y,x)


def main(whole_img_path, target_img_path, sponsor_img='sponsor/sponsor.png'):
    # 後の計算時間を考えてサイズをrate倍にする
    img_rgb = read_reshape_img(whole_img_path, rate=0.7, color=1)
    whole_img = read_reshape_img(whole_img_path, 0.7)
    target = read_reshape_img(target_img_path, 0.7)
    sponsor = read_reshape_img(sponsor_img, 0.7)

    # ターゲット画像のサイズ感と全体画像のサイズ感を初めからある程度合わせたい。
    # その他高速化手法を要検討
    print("ターゲット画像位置取得")
    threshold, res, target_w, target_h = get_template_img_pos(whole_img, target, 0.85, 0.01)
    loc_target = np.where(res >= threshold)
    loc_target = non_max_supression(loc_target)

    print("\nスポンサープロダクト位置取得")
    threshold, res_sponsor, sponsor_w, sponsor_h = get_template_img_pos(whole_img, sponsor, 0.58, 0.15)
    loc_sponsor = np.where(res_sponsor >= threshold)
    loc_sponsor = non_max_supression(loc_sponsor)
    loc_sponsor = delete_left_sponsor(loc_sponsor)

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

    cv2.imwrite('result.png', img_rgb)
    return sponsor_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole_img', type=str, default='./data/whole_renzi.png')
    parser.add_argument('--target_img', type=str, default='./data/target_renzi.png')
    opt = parser.parse_args()
    sponsor_pos = main(opt.whole_img, opt.target_img)
    print(f"\n今回の画像は{sponsor_pos}番目のスポンサープロダクトです。")
