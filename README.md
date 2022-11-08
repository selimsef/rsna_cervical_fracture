## RSNA 2022 Cervical Spine Fracture Detection
## 4th private/1st public

From https://paperswithcode.com/sota/action-classification-on-kinetics-400 it is clearly seen that the best convolutional architecture for video classifcation in 2022 is still [ir-CSN-152](https://arxiv.org/abs/1904.02811).
Even though transformers achieve higher scores on Kinetics datasets they don't perform as good on small datasets like RSNA and CSNs are the best for small sized  datasets and if one needs fast and accurate 3D segmentation and/or classification.

I used [mmaction2](https://github.com/open-mmlab/mmaction2)'s implementation of CSNs.
### Segmentation
**Model - UNet like multiclass segmentor** 
- encoder: ir-CSN-50
- decoder: standard unet decoder with nn upsampling but with [(2+1)d convolutions](https://arxiv.org/abs/1711.11248v3)
- pure 3d convolutions in decoder lead to NaNs in amp training

**Training**:
- 2x subsampling(just slice `::2`) by z axis
- 2x linear downsampled images
- memory mapping to reduce IO/CPU overhead
- AdamW + wd, cosine LR annealing
- Loss: focal-jaccard optimized loss for faster multiclass jaccard computation
- 2D augs, replay compose in Albumentations library, lightweight geometric augs + hflip

### Classification
**Model**
- 4 folds of ir(ip)-CSN-152 with global max pooling

**Target**
- multilabel 8 classes, if less than 30% of vertebra is visible then make  vertebra's label 0 and recompute overall label, otherwise keep the same as provided

**Training**

- 3 channel input (img, img, integer encoded segmentation mask 
- 2x subsampling(just slice ::2) by z axis
- using 40 slices(80 in original data) around each vertebra
- crops were resized to 256x256
- BCE loss
- target metric as validation
- AdamW + wd, cosine LR annealing
- Augmentations: 2D augs, replay compose in Albumentations library, flips, rotations, geometric. Needed to make more augs compared to segmentation pipeline

### How to reproduce 
- inference kernel on Kaggle with trained weights https://www.kaggle.com/code/selimsef/rsna-csn-segmentor-classifier
- Requirements are specified in requirements.txt
- end-to-end training is done with `run_all.sh <dataset_dir>`. This will preprocess images and store them in the same dataset directory, hence one needs additional space on the disk. Then it will train models and average checkpoints.
- Note it is for multigpu training for 4x48g gpus. 
- I could get similar results on a single 16G laptop GPU - it is much longer training time. In that case you need to change json configs for batch size/LR/crop and scripts.

