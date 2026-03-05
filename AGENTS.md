RLM(Recursive Language Model)の実装

通常のRLMと違って、PDFや画像処理を実装する。
現在のステータスは実装中。

## 構成

パッケージはドメイン単位で整理。data_model.pyにデータ構造を定義

e.g. 

- pdf/
    - data_model.py
    - convert.py
- image/
    - data_model.py  

## 規約

データモデル以外のクラスは使用せず、関数型志向。状態はデータモデルで保持する。

## Reducerパターン
複雑な状態遷移はReducerパターンで実装。

純粋関数型のreducerと、副作用ありのexecutorの組み合わせで実装

reducerはprev_stateとprev_command_resultを受け取り、next_stateとcommandを返す
executorはreducerを呼び、コマンドを実行

## テスト

- pytestを利用。
- テストもドメイン単位で配置
- 例: imageパッケージのテストは `tests/image/test_some_module.py` に配置。
- ふるまいベースで、give / when / then形式のコメントをつける。

## リンタ、フォーマッタ、テスト実行

```
make format
make lint
make typecheck
make test
```