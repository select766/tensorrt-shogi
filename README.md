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

## 将棋モデルベンチマーク

```
./cpp/multi_gpu_bench nGPU nThreadPerGPU batchSizeMin batchSizeMax profileBatchSizeMultiplier benchTime verify suppressStdout fpbit
```

* `batchSizeMin`以上`batchSizeMax`以下のバッチサイズを各iterationでランダムに選択して推論にかかる時間を測定する。
* `profileBatchSizeMultiplier`: TODO
* `nGPU`: 使用するGPU数。
* `nThreadPerGPU`: 1GPUあたりのCPUスレッド数(GPUを操作するのは同時に1スレッドのみ)。
* `verify`: 1を指定すると入出力データをファイルから読んで結果が正しいことを確認する(速度のベンチマークをする場合は確認しない"0"を指定)。
* `suppressStdout`: 1を指定するとTensorRTのログメッセージを出力しない。
* `fpbit`: 計算のビット数。8/16/32のいずれか。

典型的なベンチマーク例(8GPU):

```
./cpp/multi_gpu_bench 8 1 1 256 30 0 0 32
./cpp/multi_gpu_bench 8 1 1 256 30 0 0 16
```
