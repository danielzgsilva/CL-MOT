# CL-MOT
A contrastive learning framework for multi-object tracking

<p align="center">
  <img src="https://github.com/danielzgsilva/CL-MOT/blob/master/assets/training_cycle.png?raw=true" alt="CL-MOT training pipeline"/>
  The CL-MOT within-frame contrastive learning framework
</p>

## Abstract
The field of Multi-Object Tracking (MOT) has seen tremendous progress with the emergence of joint detection and tracking algorithms. Nevertheless, these approaches still rely on supervised learning schemas with prohibitive annotation demands. To this end, we present CL-MOT, a semi-supervised contrastive learning framework for multi-object tracking. By clustering object embeddings from different views of static frames, CL-MOT learns to discriminate between objects in a scene without relying on identity labels. This simple, yet profound pre-text learning paradigm enables one to transform any object detector into a tracker with no additional annotation labor. We evaluate the proposed approach on the MOT Challenge benchmark, finding that CL-MOT performs on par with supervised trackers and significantly outperforms other self-supervised methods. Interestingly, we even set a new state-of-the-art in re-identification metrics such as IDF1 score, suggesting that CL-MOT learns a more effective feature space than its counterparts. These results demonstrate that our within-frame contrastive learning framework can significantly reduce the annotation demands of tracking while recovering if not surpassing the performance of previous approaches.

## Features
- CL-MOT enables one to train an appearance-based tracker on static images with only bounding box labels
- We do **not** require object identity or association labels, which significantly reduces the annotation demands of tracking
- Our results match the performance of supervised trackers and significantly outperforms other self-supervised methods

## MOT Challenge Results 
Our results are gathered from the  [MOT Challenge](https://motchallenge.net/) online test server 
| Dataset | MOTA | MOTP | IDF1 | IDSW | MT | ML |
|---------|------|------|------|------|----|----|
| MOT15   |      |      |      |      |    |    |
| MOT17   |      |      |      |      |    |    |
| MOT20   |      |      |      |      |    |    |

## Usage

### Installation

### Dataset Setup

### Training

### Testing

## Acknowledgement
CL-MOT builds strongly upon the work from [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT), [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT), and [xingyizhou/CenterNet](https://github.om/xingyizhou/CenterTrack). We also use [py-motmetrics](https://github.com/cheind/py-motmetrics) for tracking evaluation and the [DCNv2](https://github.com/CharlesShang/DCNv2). deformable convolution implementation. Thank you for their great work!

