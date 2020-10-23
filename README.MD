## Amazonのスポンサープロダクト順位取得プログラム

###  必要なライブラリ
- numpy
- matplotlib
- cv2

### 使い方

##### ①Amazon画像全体のスクリーンショットを取得します  
例）  
適当なAmazonの検索画面  
https://www.amazon.co.jp/s?k=%E9%9B%BB%E5%AD%90%E3%83%AC%E3%83%B3%E3%82%B8&__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&ref=nb_sb_noss_1  
で以下のコマンドを入力(Chrome/Mac)
```
command+option+I  
command+shift+P  
「full」→return　　
```


#####  ②ターゲット画像を準備します。  
適当にスクリーンショットを取るなり、「名前をつけて画像を保存」してください。


##### ③AmazonSponsorPos.pyを実行します。
`python AmazonSponsorPos.py --whole_img=①の画像 --target_img=②の画像` 
でターゲット画像が何番目のスポンサープロダクトなのかを取得できます。  
内部で実行されている `get_sponsor_pos()`はスポンサープロダクトの番号を返します。
