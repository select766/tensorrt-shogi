# 将棋AI向けにTensorRTを動作させるサンプルコード

[TensorRT 7](https://developer.nvidia.com/tensorrt)を用いて将棋AI向けのDeep Neural Networkを実行するサンプルコードです。

DNNはResNetをベースとした畳み込み層主体のもので、画像分類と似ていますが出力が2つあるものを扱います。入力サイズ: (batchsize, 119, 9, 9)、出力サイズ: policy=(batchsize, 2187)、 value=(batchsize, 2)。様々なバッチサイズの入力に対して効率的に処理ができるよう、複数の最適化プロファイルを使い分ける機能の使用例にもなっています。

CUDA GPUが必須。Ubuntu 18.04+TensorRT 7.0.0.11で動作確認しています。コンパイル環境を整えればWindowsでも動くかもしれません。

# MNIST動作

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

# 将棋モデルベンチマーク

`multi_gpu_bench.cpp`がそのコード。

## 準備
モデルは、以下のようなPythonコードでPyTorchのものをONNX形式に変換してから使う。

```
torch.onnx.export(model, torch.randn(1, 119, 9, 9), "/path/to/output/file", export_params=True, opset_version=10,
                  verbose=True, do_constant_folding=True, input_names=["input"],
                  output_names=["output_policy", "output_value"],
                  # TensorRTでバッチサイズを可変にする際に必要
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output_policy': {0: 'batch_size'},
                                'output_value': {0: 'batch_size'}})
```

ファイルの配置

* data/trt/model.onnx
  * ONNXモデルファイル
* data/trt/inputs.bin
  * 結果評価用入力ファイル
* data/trt/policys.bin
  * 結果評価用正解出力ファイル(policy)
* data/trt/values.bin
  * 結果評価用正解出力ファイル(value)

結果評価用ファイルは、学習データを用いてPyTorchで作成。numpy行列をtofile()で保存したもの。最大バッチサイズ分だけあればよい。

## 実行
```
./cpp/multi_gpu_bench nGPU nThreadPerGPU batchSizeMin batchSizeMax profileBatchSizeRange benchTime verify suppressStdout fpbit useSerialization
```

* `batchSizeMin`以上`batchSizeMax`以下のバッチサイズを各iterationでランダムに選択して推論にかかる時間を測定する。
* `profileBatchSizeRange`: バッチサイズごとの最適化プロファイルの作成方法の指定（下記参照）
* `nGPU`: 使用するGPU数。
* `nThreadPerGPU`: 1GPUあたりのCPUスレッド数(GPUを操作するのは同時に1スレッドのみ)。
* `verify`: 1を指定すると入出力データをファイルから読んで結果が正しいことを確認する(速度のベンチマークをする場合は確認しない"0"を指定)。
* `suppressStdout`: 1を指定するとTensorRTのログメッセージを出力しない。
* `fpbit`: 計算のビット数。8/16/32のいずれか。
* `useSerialization`: 1を指定すると、エンジンデータをシリアライズして保存(`/var/tmp/multi_gpu_bench.bin`)し、2回目以降の実行ではそれを使う。

典型的なベンチマーク例(8GPU):

```
./cpp/multi_gpu_bench 8 1 1 256 1-1-256-256 30 0 0 32
./cpp/multi_gpu_bench 8 1 1 256 1-1-256-256 30 0 0 16
```

実行結果の表示は、バッチサイズごとに1回の評価にかかった時間と、そこから逆算した1秒当たりに評価できるサンプル数。1GPU当たりの値が表示される。複数GPUを合わせた性能はGPU数を掛けること。

## バッチサイズごとの最適化プロファイルについて
profileBatchSizeRange: opt1-max1-opt2-max2...

profileBatchSizeRange==10-20-100-200のとき、

* バッチサイズ1~20について、バッチサイズ10に最適化した実行計画（プロファイル）を作成
* バッチサイズ21~200について、バッチサイズ100に最適化した実行計画を作成

小さいバッチサイズに対して、大きいサイズとは別のプロファイルを作成するほうが小さいバッチサイズでの性能が高くなることが期待できる。

空文字列(`''`)を指定した場合はバッチサイズ`batchSizeMin`~`batchSizeMax`について、バッチサイズ`batchSizeMax`に最適化する（単一のプロファイルを用いる）。

# ライセンス
作者が製作した部分はMITライセンスです。

一部のソースコードは、NVIDIA社のTensorRTサンプルコード(Apache 2.0ライセンス)を改変して作られています。各ソースのヘッダーをご覧ください。
