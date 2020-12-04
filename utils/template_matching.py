import numpy as np
import cv2

def delete_not_aimed_sponsor(loc_sponsor, rate, left_sponsor_pos=300, top_sponsor_pos=800):
    # 左端に出ている広告と、右上に出ているスポンサーの説明はカウントしないよう消す
    left_id = np.argmin(loc_sponsor[1])
    if loc_sponsor[1][left_id] < left_sponsor_pos * rate:
        loc_sponsor = np.delete(loc_sponsor, left_id, axis=1)
    top_id = np.argmin(loc_sponsor[0])
    if loc_sponsor[0][top_id] < top_sponsor_pos * rate:
        loc_sponsor = np.delete(loc_sponsor, top_id, axis=1)
    return loc_sponsor


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
    return (y, x)


def get_template_pos(whole_img, target_img, threshold_goal, quality):
    count = 1
    target_org = target_img
    org_w, org_h = target_org.shape[::-1]
    threshold = 0
    thresholds = []
    while threshold < threshold_goal:
        res = cv2.matchTemplate(whole_img, target_img, cv2.TM_CCOEFF_NORMED)
        threshold = res.max()
        coef = count * 0.01
        if count % 2 == 0:
            target_img = cv2.resize(target_org, (int(org_w * (1 - coef)),
                                                 int(org_h * (1 - coef))))
        else:
            target_img = cv2.resize(target_org, (int(org_w * (1 + coef)),
                                                 int(org_h * (1 + coef))))
        print(f"\rcount={count}, threshold={threshold}", end="")
        count += 1
        if count == 50:
            return (), -1, -1
    w, h = target_img.shape[::-1]
    loc = np.where(res >= (threshold - quality))
    return loc, w, h
