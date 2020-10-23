import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse


def get_sponsor_pos(whole_img, target_img, sponsor_img='sponsor/sponsor.png'):
    # 画像の読み込み&この後の計算時間を考えてサイズを1/4へ
    img_rgb = cv2.imread(whole_img)
    img_rgb = cv2.resize(img_rgb, (int(img_rgb.shape[1] * 0.5), int(img_rgb.shape[0] * 0.5)))
    img_rgb = img_rgb[:, int(img_rgb.shape[1] * 0.24):]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(target_img, 0)
    template = cv2.resize(template, (int(template.shape[1] * 0.5), int(template.shape[0] * 0.5)))
    template_base = template
    w, h = template.shape[::-1]
    sponsor = cv2.imread(sponsor_img, 0)
    sponsor = cv2.resize(sponsor, (int(sponsor.shape[1] * 0.5), int(sponsor.shape[0] * 0.5)))
    threshold = 0

    # なぜかターゲット画像を取得した際に実際と違うサイズで取得される場合があるっぽいので、
    # ターゲット画像の大きさを変えながら一致するタイミングを探す
    # だいぶ雑なので、もし取り損ねていることがあったら教えてください。
    count = 1
    while threshold < 0.85:
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = res.max() - 0.01
        print(threshold)
        coef = count * 0.01
        w, h = template.shape[::-1]
        if count % 2 == 0:
            template = cv2.resize(template_base, (int(template_base.shape[1] * (1 - coef)),
                                                  int(template_base.shape[0] * (1 - coef))))
        else:
            template = cv2.resize(template_base, (int(template_base.shape[1] * (1 + coef)),
                                                  int(template_base.shape[0] * (1 + coef))))
        count += 1
    loc = np.where(res >= threshold)
    pos_w, pos_h = loc[1][0], loc[0][0]
    cv2.rectangle(img_rgb, (pos_w, pos_h), (pos_w + w, pos_h + h), (0, 0, 255), 2)

    # スポンサープロダクトの位置を取得
    res_sponsor = cv2.matchTemplate(img_gray, sponsor, cv2.TM_CCOEFF_NORMED)
    loc_sponsor = np.where(res_sponsor >= 0.8)
    distance = 10000
    sponsor_pos = 100

    # スポンサープロダクトの何番目かを計算
    for index, pt in enumerate(zip(*loc_sponsor[::-1])):
        cv2.rectangle(img_rgb, pt, (pt[0] + sponsor.shape[1], pt[1] + sponsor.shape[0]), (0, 0, 255), 2)
        temp_distance = abs(pt[0] - pos_w) + abs(pt[1] - (pos_h + h))
        if temp_distance < distance:
            distance = temp_distance
            sponsor_pos = index + 1

    cv2.imwrite('res.png', img_rgb)
    return sponsor_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole_img', type=str, default='renzi.png')
    parser.add_argument('--target_img', type=str, default='renziTarget.png')
    opt = parser.parse_args()
    sponsor_pos = get_sponsor_pos(opt.whole_img, opt.target_img)
    print(f"今回の画像は{sponsor_pos}番目のスポンサープロダクトです。")
