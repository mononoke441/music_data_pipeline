# PANNs runtime subset

This directory vendors the minimal Python source required by the production
PANNs MobileNetV1 music gate:

- `pytorch/models.py`
- `pytorch/pytorch_utils.py`

The source comes from
[`qiuqiangkong/audioset_tagging_cnn`](https://github.com/qiuqiangkong/audioset_tagging_cnn)
and remains under its MIT license; see `LICENSE.MIT`.

Model weights are not stored in Git. Download and verify the production
MobileNetV1 checkpoint with:

```bash
bash scripts/download_gate_assets.sh all
```

`MusicToolsPipeline/checkpoints/fast_gate_config.json` pins the expected model
SHA256 and the exact runtime-source fingerprint.
