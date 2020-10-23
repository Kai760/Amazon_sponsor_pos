### 使い方

①Amazon画像全体のスクリーンショットを取得します
```
command+option+I  
command+shift+P  
「full」→return　　
```
仮に今回は「renzi.png」を取得したとします。


②ターゲット画像を準備します。  
今回は仮に「renzi_target.png」を取得したとします

③amazon_sponsor_pos.py内にある  
 `get_sponsor_pos(全体画像, ターゲット画像)` を実行してください。
返り値はスポンサープロダクトの番号です。
