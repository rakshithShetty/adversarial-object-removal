# adversarial-object-removal
Code base for our paper " Adversarial Scene Editing: Automatic Object Removal from Weak Supervision" appearing in NIPS 2018.

# Todo
- [x] Upload dataset file and pre-trained model
- [ ] Usage instruction
- [ ] port to python 3.6
- [ ] port to pytorch 1.0

# Downloads
* [Pre-trained model](https://datasets.d2.mpi-inf.mpg.de/rakshith/object_removal_nips/checkpoint_stargan_coco_fulleditor_LowResMask_pascal_RandDiscrWdecay_wgan_30pcUnion_noGT_imnet_V2_msz32_ftuneMask_withPmask_L1150_tv_nb4_styleloss3k_248_1570.pth.tar) - Weakly supervised removal model for 20 pascal object categories trained on COCO dataset.
* [COCO dataset file](https://datasets.d2.mpi-inf.mpg.de/rakshith/object_removal_nips/datasetBoxAnn_80pcMaxObj_mrcnnval.json) - Single json file with metadata and annotations for the COCO dataset
* [Poster](https://datasets.d2.mpi-inf.mpg.de/rakshith/object_removal_nips/NIPS2018_poster.pdf)

# Bibtex
If you find this code useful in your work, please cite the paper.
```
@inproceedings{shetty_neurips2018,
TITLE = {Adversarial Scene Editing: Automatic Object Removal from Weak Supervision},
AUTHOR = {Shetty, Rakshith and Fritz, Mario and Schiele, Bernt},
PUBLISHER = {Curran Associates},
YEAR = {2018},
BOOKTITLE = {Advances in Neural Information Processing Systems 31},
PAGES = {7716--7726},
ADDRESS = {Montr{\'e}al, Canada},
}
```

# Acknowledgements
Lot of the code structure is borrowed from the Stargan repository (https://github.com/yunjey/stargan)
