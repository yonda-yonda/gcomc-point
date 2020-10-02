
# GCOM-C point
GCOM-Cの観測データを時空間データとしてElasticsearchに入れるサンプル

## 対象
* L2陸域
* L2海洋

## 実行
1. 対象ファイルをsourceにダウンロード
1. `docker-compose build`
1. `docker-compose up -d`
1. http://localhost:8888/ にアクセス
1. work配下のノートブックを実行
1. 終わったら`docker-compose down`

## ライセンス
自由に改変して利用してください。