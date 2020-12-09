# Amazonのスポンサープロダクト順位取得プログラム

###  必要なライブラリ
- numpy
- cv2

### 使い方

#### ①Amazon画像全体のスクリーンショットを取得します  
例）  
適当なAmazonの検索画面  
https://www.amazon.co.jp/s?k=%E9%9B%BB%E5%AD%90%E3%83%AC%E3%83%B3%E3%82%B8&__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&ref=nb_sb_noss_1  
で以下のコマンドを入力しスクリーンショットを入手　　
(Chrome/Macなら以下の通り)
```
command+option+I  
command+shift+P  
「full」→return　　
```
(推奨)もしくはchromeから以下のキャプチャソフトを使用  
"GoFullPage"  
https://chrome.google.com/webstore/detail/gofullpage-full-page-scre/fdpohaocaechififmbbbbbknoalclacl

####  ②ターゲット画像を準備します。  
適当にスクリーンショットを取るなり、「名前をつけて画像を保存」してください。


#### ③AmazonSponsorPos.pyを実行します。
`python amazon_sponsor_pos.py --whole_img=①の画像 --target_img=②の画像` 
でターゲット画像が何番目のスポンサープロダクトなのかを取得できます。  
内部で実行されている `main()`はスポンサープロダクトの番号を返します。

#### 現在の決め打ちポイント
基準長=全体画像の横幅であるときに
- ターゲット画像の最大長 = 基準長*0.25
- 画像探し打ち切りライン = ターゲット画像の(0.5~1.5)倍を探索し終えたら
- ターゲット画像の類似閾値 = 0.95
- スポンサー画像の類似閾値 = 0.7
- 左にあるスポンサー広告位置 = 基準長*0.15
- 右上にあるスポンサー画像位置 = 基準長*0.15
- 被り矩形消去の基準=基点の距離が30以内の矩形