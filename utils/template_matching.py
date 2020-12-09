import numpy as np
import cv2


def delete_not_aimed_sponsor(loc_sponsor, ref_length, left_sponsor_rate=0.15, right_sponsor_rate=0.85):
    # 左端に出ているバナー広告の「スポンサー」と、右上に出ている「スポンサー」はカウントしないよう消す
    left = ref_length * left_sponsor_rate
    right = ref_length * right_sponsor_rate
    loc_return = (np.delete(loc_sponsor[0], np.where((loc_sponsor[1] < left) | (right < loc_sponsor[1]))),
                  np.delete(loc_sponsor[1], np.where((loc_sponsor[1] < left) | (right < loc_sponsor[1]))))

    return loc_return


def non_max_supression(loc):
    # ある矩形の近くにある矩形は同一のものを囲っているとみなす
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
    return (y, x)

def reshape_template_img(count, target_org):
    # テンプレートマッチングで見つからなかった時に、画像のサイズをちょっと変える
    org_w, org_h = target_org.shape[::-1]
    coef = count * 0.01
    if count % 2 == 0:
        target_img = cv2.resize(target_org, (int(org_w * (1 - coef)),
                                             int(org_h * (1 - coef))))
    else:
        target_img = cv2.resize(target_org, (int(org_w * (1 + coef)),
                                             int(org_h * (1 + coef))))
    return target_img


def get_template_pos(whole_img, target_img, threshold_goal, quality):
    count = 0
    target_org = target_img.copy()
    threshold = 0
    # 閾値を超えるまで画像サイズを変えながらテンプレートマッチングを実行
    while threshold < threshold_goal:
        target_img = reshape_template_img(count, target_org)
        res = cv2.matchTemplate(whole_img, target_img, cv2.TM_CCOEFF_NORMED)
        threshold = res.max()
        print(f"\rcount={count}, threshold={threshold}", end="")
        count += 1
        # 50回（元の0.5~1.5倍まで）試してなかったらないものとする。ここの調整は要検討
        if count == 50:
            return (), -1, -1
    w, h = target_img.shape[::-1]
    loc = np.where(res >= (threshold - quality))
    return loc, w, h
