# tensorrt-shogi
将棋AI向けにTensorRTを動作させる

## MNIST動作

プロジェクトルートディレクトリに並列に`TensorRT-7.0.0.11`を展開

```
python -m venv venv
source ./venv/bin/activate
# モデル学習・ONNX出力
python -m pttrain.mnist_first
# テストデータ出力
python -m pttrain.export_mnist
```

```
cd cpp && make && cd ..
```

実行
```
LD_LIBRARY_PATH=../TensorRT-7.0.0.11/lib ./cpp/mnist
```
