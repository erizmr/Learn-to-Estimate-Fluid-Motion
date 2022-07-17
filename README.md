# Learn-to-Estimate-Fluid-Motion
Learning to Estimate and Refine Fluid Motion with Physical Dynamics (ICML 2022)
[[paper]](https://arxiv.org/pdf/2206.10480.pdf)

```BibTeX
@InProceedings{zhang22learning,
  title = 	 {Learning to Estimate and Refine Fluid Motion with Physical Dynamics},
  author =       {Zhang, Mingrui and Wang, Jianhong and Tlhomole, James B and Piggott, Matthew},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  year = 	 {2022}
}
```


## Installation

```bash
pip install -r requirements.txt
```

## Dataset

```bash
sh download_data.sh
```

## Pretrained Model

```bash
sh download_pretrained_model.sh
```

## Run the pretrained model

```py
python3 demo.py --config-file pretrained_model/config_pretrained.yaml
```

## Acknowledgement

The training framework is largely adapted from the [Detectron2](https://github.com/facebookresearch/detectron2).

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```