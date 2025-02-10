# [NeurIPS 2024] Efficient Large Multi-modal Models via Visual Context Compression

The folder includes the implementation of LLaVolta for Efficient Large Language and Vision Assistant. 

<p>
<img src="staging2.png" alt="teaser" width=90% height=90%>
</p>

```bibtex
@inproceedings{chen2024efficient,
  title={Efficient large multi-modal models via visual context compression},
  author={Chen, Jieneng and Ye, Luoxin and He, Ju and Wang, Zhao-Yang and Khashabi, Daniel and Yuille, Alan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

## Instantiation of LLaVolta schemes:

<img width="841" alt="image" src="https://github.com/Beckschen/LLaVolta/assets/30471421/62831a80-1e7c-4a07-b5e2-38296c3b88cd">

## Accelerate and Boost LLaVA:

<img width="876" alt="image" src="https://github.com/Beckschen/LLaVolta/assets/30471421/35b903ac-15ba-48be-8b9c-7823af0a1dc7">

## Accelerate and Boost VideoLLaVA:

<img width="840" alt="image" src="https://github.com/Beckschen/LLaVolta/assets/30471421/e010cd53-16d9-44bf-a281-58cedca0600c">


## Install
*Note: code is developed based on Ubuntu 20.04/22.04. CUDA=12.1*
Our code is developed based on LLaVA, the installation is very similar to original repo of LLaVA:
1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/Beckschen/LLaVolta
cd LLaVolta
```

2. Install Package
```Shell
conda create -n llavolta python=3.10 -y
conda activate llavolta
pip install --upgrade pip 
pip install -e .
```

3. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
cd llava/eval
tar xvf table.tar
cd ../..
```

## Efficient Training
1. Download the training data for both pretraining and fine-tuning from the original LLaVA repository.
2. Set the necessary path variables: `ROOT_DATA`, `ROOT_WEIGHT`, and `ROOT_LOG` (optional).
3. Begin training using the [scripts](https://github.com/Beckschen/LLaVolta/scripts/v1_5). We provide four examples: 4stage, heavy_compression, light_compression, and reproduce.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/train-$NAME.sh
```
## Evaluation
Running scripts under scripts/v1_5/eval/$NAME, where NAME is the name of checkpoint's name. We provide four example: 4stage, heavy_compression, light_compression, reproduce.

For all scripts we provided, please first fill up necessary path variables: **ROOT_DATA**, **ROOT_WEIGHT**, **ROOT_LOG**(optional)


### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `$ROOT_DATA/eval/vqav2`.
2. Multi-GPU inference.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/vqav2.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission).

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `$ROOT_DATA/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Multi-GPU inference.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/gqa.sh
```

### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `$ROOT_DATA/eval/vizwiz`.
2. Single-GPU inference.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/vizwiz.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission): `$ROOT_DATA/eval/vizwiz/answers_upload`.

### ScienceQA

1. Under `$ROOT_DATA/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `$ROOT_DATA/eval/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/textvqa.sh
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `$ROOT_DATA/eval/pope`.
2. Single-GPU inference and evaluate.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/pope.sh
```

### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `$ROOT_DATA/eval/MME`.
4. Single-GPU inference and evaluate.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/mme.sh
```

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `$ROOT_DATA/eval/mmbench`.
2. Single-GPU inference.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/mmbench.sh
```
3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `$ROOT_DATA/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `$ROOT_DATA/eval/mmbench`.
2. Single-GPU inference.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/mmbench_cn.sh
```
3. Submit the results to the evaluation server: `$ROOT_DATA/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


### SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images and the videos. Put images under `$DATA_ROOT/eval/seed_bench/SEED-Bench-image`. Note that we only use image subset to evaluate LLaVolta
3. Multiple-GPU inference and evaluate.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/seed.sh
```

### LLaVA-Bench-in-the-Wild

1. Extract contents of [`llava-bench-in-the-wild`](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) to `$ROOT_DATA/eval/llava-bench-in-the-wild`.
2. Single-GPU inference and evaluate.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/llavabench.sh
```

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `$ROOT_DATA/eval/mmvet`.
2. Single-GPU inference.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/mmvet.sh
```
3. Evaluate the predictions in `$ROOT_DATA/eval/mmvet/results` using the official jupyter notebook.

 

## Citing LLaVolta
```bibtex
@inproceedings{chen2024efficient,
  title={Efficient large multi-modal models via visual context compression},
  author={Chen, Jieneng and Ye, Luoxin and He, Ju and Wang, Zhao-Yang and Khashabi, Daniel and Yuille, Alan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```


## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Vicuna](https://github.com/lm-sys/FastChat)

Luoxin Ye (@feiyu12138) is the primary contributor to most of the codebase, including both the training and evaluation pipelines. We have archived these projects here, maintaining a clean and organized code style.
