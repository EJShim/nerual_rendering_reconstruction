# Python 3.9 환경 사용
```
conda create -n fastopt2 python=3.9
```

# torch 1.13.1
CUDA 11.6 으로 설치

```
# CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# or conda
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

```



# Pytorch3d windows
source 에서 설치, vscode compiler 사용가능한 터미널
```
$env:DISTUTILS_USE_SDK = 1 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```