import os
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary


# このままだと写っている部分しか取れない！全画面はどうやる？？
def get_screenshot():
    # File Name
    filename = "ss.png"

    # set driver and url
    # driver = webdriver.Chrome('./chromedriver')
    driver = webdriver.Chrome()
    url = "https://www.amazon.co.jp/s?k=%E3%83%AF%E3%82%A4%E3%83%A4%E3%83%AC%E3%82%B9%E3%82%A4%E3%83%A4%E3%83%9B%E3%83%B3&__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&ref=nb_sb_noss_2"
    driver.get(url)

    # get w x h and screenshot
    w = driver.execute_script("return document.body.scrollWidth;")
    h = driver.execute_script("return document.body.scrollHeight;")
    driver.set_window_size(w, h)
    driver.save_screenshot(filename)
    driver.quit()


if __name__ == '__main__':
    pass
