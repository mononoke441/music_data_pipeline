---
license: cc-by-nc-4.0
language:
- en
- zh
pipeline_tag: audio-classification
tags:
- music
---

# MuQ & MuQ-MuLan

<div>
  <a href='#'><img alt="Static Badge" src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white"></a>
  <a href='https://arxiv.org/abs/2501.01108'><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2501.01108-%23b31b1b?logo=arxiv&link=https%3A%2F%2Farxiv.org%2F"></a>
  <a href='https://huggingface.co/OpenMuQ'><img alt="Static Badge" src="https://img.shields.io/badge/huggingface-OpenMuQ-%23FFD21E?logo=huggingface&link=https%3A%2F%2Fhuggingface.co%2FOpenMuQ"></a>
  <a href='https://pytorch.org/'><img alt="Static Badge" src="https://img.shields.io/badge/framework-PyTorch-%23EE4C2C?logo=pytorch"></a>
  <a href='https://pypi.org/project/muq'><img alt="Static Badge" src="https://img.shields.io/badge/pip%20install-muq-green?logo=PyPI&logoColor=white&link=https%3A%2F%2Fpypi.org%2Fproject%2Fmuq"></a>
</div>


This is the official repository for the paper *"**MuQ**: Self-Supervised **Mu**sic Representation Learning
 with Mel Residual Vector **Q**uantization"*. For more detailed information, we strongly recommend referring to https://github.com/tencent-ailab/MuQ and the [paper]((https://arxiv.org/abs/2501.01108)). 

In this repo, the following models are released:

- **MuQ**(see [this link](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter)): A large music foundation model pre-trained via Self-Supervised Learning (SSL), achieving SOTA in various MIR tasks.
- **MuQ-MuLan**(see [this link](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)): A music-text joint embedding model trained via contrastive learning, supporting both English and Chinese texts.


## Usage

To begin with, please use pip to install the official `muq` lib, and ensure that your `python>=3.8`:
```bash
pip3 install muq
```



To extract music audio features using **MuQ**:
```python
import torch, librosa
from muq import MuQ

device = 'cuda'
wav, sr = librosa.load("path/to/music_audio.wav", sr = 24000)
wavs = torch.tensor(wav).unsqueeze(0).to(device) 

# This will automatically fetch the checkpoint from huggingface
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
muq = muq.to(device).eval()

with torch.no_grad():
    output = muq(wavs, output_hidden_states=True)

print('Total number of layers: ', len(output.hidden_states))
print('Feature shape: ', output.last_hidden_state.shape)

```



Using **MuQ-MuLan** to extract the music and text embeddings and calculate the similarity:
```python
import torch, librosa
from muq import MuQMuLan

# This will automatically fetch checkpoints from huggingface
device = 'cuda'
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan = mulan.to(device).eval()

# Extract music embeddings
wav, sr = librosa.load("path/to/music_audio.wav", sr = 24000)
wavs = torch.tensor(wav).unsqueeze(0).to(device) 
with torch.no_grad():
    audio_embeds = mulan(wavs = wavs) 

# Extract text embeddings (texts can be in English or Chinese)
texts = ["classical genres, hopeful mood, piano.", "一首适合海边风景的小提琴曲，节奏欢快"]
with torch.no_grad():
    text_embeds = mulan(texts = texts)

# Calculate dot product similarity
sim = mulan.calc_similarity(audio_embeds, text_embeds)
print(sim)
```

## Model Checkpoints

| Model Name | Parameters | Data | HuggingFace🤗 |
| ----------- | --- | ---  | ----------- |
| MuQ    | ~300M  | MSD dataset | [OpenMuQ/MuQ-large-msd-iter](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter)       |
| MuQ-MuLan  | ~700M | music-text pairs | [OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)       |

**Note**: Please note that the open-sourced MuQ was trained on the Million Song Dataset. Due to differences in dataset size, the open-sourced model may not achieve the same level of performance as reported in the paper.

## License

The code is released under the MIT license.

The model weights (MuQ-large-msd-iter, MuQ-MuLan-large) are released under the CC-BY-NC 4.0 license. 

## Citation

```
@article{zhu2025muq,
      title={MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization}, 
      author={Haina Zhu and Yizhi Zhou and Hangting Chen and Jianwei Yu and Ziyang Ma and Rongzhi Gu and Yi Luo and Wei Tan and Xie Chen},
      journal={arXiv preprint arXiv:2501.01108},
      year={2025}
} 
```
