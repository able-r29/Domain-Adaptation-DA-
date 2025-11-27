# Usage  
## Make  
※必須ではない  

実験用のディレクトリを作成して現在のコードをコピーするプログラム  
これにより、実験結果とその時のソースコードが完全に1対1対応されるため、再現がしやすいはず  

### Usage  
 - root, r : str, optional  
   出力先ディレクトリ  
 - post, p : str, optional  
   出力名称の最後に付ける文字列  
   メモとして使える  

コマンド例  
```
python3 make.py -r ../results -p test_exp
```
これにより`../results/2022_02_20__14_53_32__test_exp`が作成される  
さらにその中に`code`フォルダが作成され、現在のコードがコピーされている  


## Trainer  
学習を実行するコード  
実験条件ファイルとFold番号、デバイスを指定して学習を実行する  

### Usage  
 - config, c : str  
   実験条件ファイルのパス  
 - fold, p : int  
   Fold番号
 - device, d : str  
   実行デバイス

コマンド例  
```
python3 trainer.py -c ./config/wtp_2021_10_13.json -f 0 -d cuda:0
```
これにより修論5章の`ResNet18+Metadata+Weighted Metric Learning`によりFold0番の学習がGPU0番で行われる  
実験結果は`../0`に出力される  


## Predict  
学習が終わった実験結果を用いて推論を行うコード  

### Usage  
 - target, t : str  
   テスト用のFold分割パターンファイルのパス
 - device, d : str  
   実行デバイス
 - fold, f : int, optional  
   推論するFold  
   複数指定可能  

コマンド例  
```
python3 predict.py -t /path/to/test.json -d cuda:0
```
これにより`path/to/test.json`に記述されているテスト画像に対する推論がGPU0番で全てのFoldに対して行われる  

### Output fotmat  
推論により、各Foldには`predict.pickle`と`predict_appendix.pickle`が作成される  
このうち`predict_appendix.pickle`はモデル毎に出力が異なるうえ、特に考察には使用しないので放置する  

`predict.pickle`のフォーマットは以下の通り  
```
predict.pickle
|-y
| |-y_1
| |-y_2
| | ...
|
|-t
| |-t_1
| |-t_2
| | ...
--m
  |-m_1
  |-m_2
  | ...
```
ここで`y_*`は各画像の推論結果でSoftmax前の値、`t_*`は正解ラベル、`m_*`はメタデータを表す  

サンプル  
```
>>> y, t, m = io.load_pickle('predict.pickle')
>>> len(y)
43915
>>> len(t)
43915
>>> len(m)
43915
>>> type(y[0])
<class 'numpy.ndarray'>
>>> y[0].shape
(59,)
>>> y[0]
array([ 2.7900336 ,  0.16271582, -5.2312503 , -0.4388231 , -3.7684705 ,
       -0.11546749, -2.7992117 , -2.975155  , -6.1342187 , -0.4677915 ,
       -2.3566597 ,  4.6844435 , -3.3068397 ,  0.74978316,  1.3799937 ,
        0.84087616, -2.7529318 , -1.4048052 , -4.0819416 , -2.4507802 ,
       -3.882567  , -5.8493104 , -3.803021  , -5.4050646 , -4.283692  ,
       -5.1329956 , -1.7109826 , -2.798207  , -5.665183  , -6.4110456 ,
       -5.6932178 , -2.19661   , -1.425454  , -3.4189646 , -1.4484177 ,
       -3.4734428 , -3.9789941 , -2.6034868 , -5.2329693 , -0.42309055,
       -2.389802  , -3.2000718 , -4.693863  , -2.6417956 , -5.4200807 ,
       -5.103642  , -1.5696979 , -5.966878  , -1.5307746 , -3.002359  ,
       -2.3020384 , -3.4032059 , -2.6469283 , -3.45068   ], dtype=float32)
>>> type(t[0])
<class 'numpy.int64'>
>>> t[0].shape
()
>>> t[0]
0
>>> type(m[0])
<class 'dict'>
>>> m[0]
{'ID': '1328', 'date_photo': '2009/2/2', 'univ_ID': 'KO2', 'date_birth': '', 'age': ['70'], 'sex': [''], 'race': ['Mongoloid'], 'bool_patho': [False], 'bool_inidiag': [True], 'bool_fidiag': [True], 'inidiag': ['basal cell carcinoma (rodent ulcer_basiloma)'], 'fidiag': ['basal cell carcinoma (rodent ulcer_basiloma)'], 'part': ['Face'], 'bool_dermos': [False], 'filename': '1328_2009_2_2_KO2_4.jpg', 'csv_src': 'NSDD-20190917/KO/tukuba_text_KO2_201902.csv', 'jpg_src': 'NSDD-20190917/KO/tukuba_image_KO2_201902/1328/1328_2009_2_2_KO2_4.jpg', 'UCSV': 'NSDD-20190917__KO__tukuba_text_KO2_201902.csv', 'UJPG': 'NSDD-20190917__KO__tukuba_image_KO2_201902__1328__1328_2009_2_2_KO2_4.jpg', 'HASH': 'ZVlqc19scGdYaHx/d2tuZWNzfn99cmxdaW53fnp5aFVkb4WYgndfV151h5GEdVRbUHR2eHtlRmVA\nVmNkX09GYA==\n', 'UID': 'NSDD-20190917__KO__tukuba_text_KO2_201902.csv___1328', 'LABEL': 0, 'CASE': 'NSDD-20190917__KO__tukuba_text_KO2_201902.csv___1328___0'}
>>>
```


## Analyze  
いくつかの依存ファイルをコピーしたため、もしかしたらエラーが出るかも  
ファイルが見つからない旨のエラーが出た場合は早めに質問するか、HDDから探すこと  

### Accuracy  
正解率を計算するコード  
2種類あり、1つの実験結果に対して細かく出力するものと、複数の結果をまとめて出力するものがある  
使い分けるのが面倒だけど、書き直すのも面倒だった  

#### calc_accuracy.py  
1つの実験について5Foldの各結果やレベル毎のクラス単位正解率を求める際にはこっちを使用する  

##### Usage  
 - src, s : str  
   実験結果のディレクトリ
   複数指定可能

コマンド例  
```
python3 calc_accuracy.py -s ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0 ../../submit_result/2022_01_02__00_33_26_triplet_a02_g1
```
これにより各実験結果のディレクトリに`analyze/accuracies/total.csv`が作成される  
ファイル内にはWeighted avgとMacro acg、画像単位と症例単位、レベル123の各Foldの正解率と平均、標準偏差、クラス単位の正解率が出力される  
クラス単位の方は画像単位レベル1～３、症例単位レベル1～3がくっついて出力されるので見づらいが、クラス番号が書いてあるので見分けてください  

#### compare_accuracy.py  
複数の実験を比較するために使用する  

##### Usage  
 - src, s : str
   実験結果のディレクトリ
   複数指定可能
 - dst, d : str
   出力先パス
 - folds, f : int, optional
   計算するFold番号
   複数指定可能

コマンド例
```
python3 compare_accuracy.py -s ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0 ../../submit_result/2022_01_02__00_33_26_triplet_a02_g1 -d ./result.csv
```
これにより`result.csv`に実験結果がまとめられて出力される  
フォーマットは`calc_accuracy.py`とほぼ同じなので省略  

### Test  
実験結果の統計検定を行う  
コードは`calc_mcnemar.py`  

#### Usage  
 - src, s : str
   実験結果のディレクトリ
   複数指定可能
 - dst, d : str
   出力先パス
 - folds, f : int, optional
   計算するFold番号
   複数指定可能

コマンド例  
```
python3 calc_mcnemar.py -s ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0 ../../submit_result/2022_01_02__00_33_26_triplet_a02_g1 -d ./test.csv
```
これにより`test.csv`に検定結果が出力される  
CSVファイル内には全ての実験結果間で総当たりに検定を行った結果のp値を記載してある  
以下の様な場合には、`d:\Experiment\NSDD\results_ptlr\2022_01_02__00_28_17_triplet_a02_g0`と`d:\Experiment\NSDD\results_ptlr\2022_01_02__00_34_14_triplet_a02_g4`との間で検定を行っており、各行がFold番号によるペア、各列が画像単位と症例単位のレベル1~3の条件で検定を行っている  
```
d:\Experiment\NSDD\results_ptlr\2022_01_02__00_28_17_triplet_a02_g0
d:\Experiment\NSDD\results_ptlr\2022_01_02__00_34_14_triplet_a02_g4
6.69E-05	7.24E-06	2.14E-05	0.872695423	0.304706671	0.778527639
1.40E-10	4.89E-13	1.29E-11	0.609603486	0.05206427	0.026232864
1.46E-06	9.08E-09	1.69E-13	0.713336187	0.582663285	0.093607191
0.000142803	6.20E-05	1.32E-08	0.420743982	0.064988642	0.94182763
0.6267529	0.777824852	4.77E-06	0.15161014	0.215298253	0.141021281
```

ただしこれだと見辛過ぎるので、p値が0.01と0.05を下回っているFold数がいくつあるのかをカウントした画像も出力する（`test.csv_p001.png`と`test.csv_p005.png`）  
画像内には6つのヒートマップが掛かれているが、上段が画像単位、下段が症例単位で、左からレベル1～3の結果になる  
p値が0.01あるいは0.05を下回ったFoldの数を色で表している  


### Confusion matrix  
混同行列を出力するコード  
`plot_confusion_matrix.py`を使用する  

#### Usage  
 - src, s : str
   実験結果のディレクトリ
 - fold, f : int, optional
   使用するFold番号
   複数指定可能

コマンド例
```
python3 plot_confusion_matrix.py -s ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0
```
これにより実験結果のディレクトリに`confusion`というフォルダが作成され、混同行列の画像が保存される  
画像のファイル名に`Image`と付いている場合は画像単位、`Case`では症例単位になる  
`Normed`と付いている場合には正解クラスの画像枚数で除算し正規化している  
最後の数字は計算に使用したFold番号を表す  


### Compare confusion matrix  
混同行列を比較するコード  
`compare_confusion_matrix.py`を使用する  

#### Usage  
 - base, b : str
   実験結果のディレクトリ
   混同行列の差分は`target - base`の値が出力される
 - target, t : str
   実験結果のディレクトリ
 - fold, f : int, optional
   使用するFold番号
   複数指定可能

コマンド例
```
python3 compare_confusion_matrix.py -b ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0 -t ../../submit_result/2022_01_02__00_33_26_triplet_a02_g1
```
これにより実験結果のディレクトリに`compare_confusion`というフォルダが作成され、混同行列の差分画像が保存される  
ファイル名のフォーマットは混同行列と同じなので省略  


### Class-wise metadata  
クラス単位のメタデータ頻度を計算する  
コードは`plot_meta_frequent.py`  

#### Usage  
 - src, s : str
   学習データのFold分割パターンファイルのパス
 - dst, d : str, optional
   CSVでの出力先

コマンド例  
```
python3 plot_meta_frequent.py -s /path/to/train.json -d metadata.csv
```
これにより`/path/to/train.json`を読み込んでメタデータの頻度をヒートマップでプロットする  
`dst`キーワードを指定していた場合はその出力先にヒートマップの値を出力する  


### Metadata classification accuracy  
メタデータ分類正解率を計算する  
学習済みパラメータを指定すると重みの類似度も表示する  
コードは`plot_meta_similarity.py`  

#### Usage  
 - train, t : str  
   学習用のFold分割パターンのJSONファイルパス
 - model, m : str, optional  
   学習済みResNet18のパス  
 - dst, o : str, optional  
   メタデータ分類正解率の数値をCSVで出力  

コマンド例
```
python3 plot_meta_similarity.py -t /path/to/train.json -m /path/to/trained.pt
```
これにより`/path/to/train.json`を読み込んでメタデータ分類正解率をヒートマップでプロットする  
学習済みモデルのパスも指定することで全結合層のコサイン類似度も示し、fruchterman-reingoldアルゴリズムによる関係性の可視化も行う  
Figure1がメタデータ分類正解率  
Figure2が重みのコサイン類似度
Figure3がメタデータ分類正解率の力学モデル可視化  
Figure4がコサイン類似度の力学モデル可視化  



### Improved image  
手法間で向上が確認された画像を取得する  
コードは`predict_difference.py`  

#### Usage  
 - base, b : str
   実験結果のディレクトリ
 - target, t : str
   実験結果のディレクトリ
 - img, i : str
   画像一式が置かれているディレクトリパス
   コンテナ起動時に毎回展開しているやつと同じく展開済みのものを指定
 - dst, d : str
   出力先ディレクトリパス

コマンド例
```
python3 predict_difference.py -b ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0 -t ../../submit_result/2022_01_02__00_33_26_triplet_a02_g1 -i /path/to/img -d /path/to/dst
```
これにより`2022_01_02__00_33_26_triplet_a02_g1`と`2022_01_02__00_28_17_triplet_a02_g0`のFold毎正解数が3以上異なる画像をピックアップし、3Fold以上改善したものを`dst/improve`に、悪化したものを`dst/worsen`に保存する  
各画像は入力時の前処理がなされたものと、5Foldの推論結果の確信度平均が高い上位10クラスに対する確信度分布を示している  



### Silhouette coefficient  
シルエット係数を計算する  

#### Intermidiate feature  
計算のために、まず中間特徴の算出を行う必要がある  
コードは`feature.py`  

##### Usage  
 - target, t : str  
   テスト用のFold分割パターンファイルのパス
 - device, d : str  
   実行デバイス
 - fold, f : int, optional  
   推論するFold  
   複数指定可能  

Predictと同一の実行方法なので、詳細は割愛

##### Output format  
推論されると各Foldのディレクトリに`predict_feature.pickle`が作成される  
フォーマットはPredictとほぼ同一であるが、各クラスのlogitの代わりにGlobal avg pool後の特徴ベクトルが出力される

#### Calculate  
シルエット係数の他にいくつかの評価値を試していたので、係数の計算とプロットが別のコードになっている  
そのため先ずはシルエット係数の計算を行う  
コードは`calc_silhouette.py`  

##### Usage  
 - src, s : str
   実験結果のディレクトリ
   複数指定可能

コマンド例
```
python3 calc_silhouette.py -s ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0 ../../submit_result/2022_01_02__00_33_26_triplet_a02_g1
```
これにより各実験結果ディレクトリに`silhouette`というフォルダが作成され、シルエット係数がCSVファイルに保存される  

#### Plot  
計算したシルエット係数をヒートマップにプロットする  
コードは`plot_silhouette.py`  

##### Usage  
 - src, s : str
   実験結果のディレクトリ

コマンド例
```
python3 plot_silhouette.py -s ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0
```
これによりシルエット係数のCSVファイルがある場所に画像ファイルが保存される  

#### Compare  
シルエット係数を比較する  
コードは`plot_silhouette_difference.py`  

##### Usage  
 - base, b : str
   実験結果のディレクトリ
 - target, t : str
   実験結果のディレクトリ

コマンド例
```
python3 plot_silhouette_difference.py -b ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0 -t ../../submit_result/2022_01_02__00_33_26_triplet_a02_g1
```
これにより値域を共通にしたシルエット係数と、`target - base`の差分が表示される  


### Probability  
画像に対する5Foldの確信度を表示する  
コードは`plot_probability.py`  

#### Usage  
 - src, s : str
   実験結果のディレクトリパス
 - img, i : str
   画像ディレクトリパス
 - index, c : int
   対象のクラスを指定

コマンド例
```
python3 plot_probability.py -s ../../submit_result/2022_01_02__00_28_17_triplet_a02_g0 -i /path/to/img -c 0
```
これにより実験結果のディレクトリに`probability`というフォルダが作成され、基底細胞癌の画像に対する推論結果が保存される  
