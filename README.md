# CL-MOT
A contrastive learning framework for multi-object tracking

![Alt text](assets/training_cycle.png?raw=true "Title")

## Abstract
The field of Multi-Object Tracking (MOT) has seen tremendous progress with the emergence of joint detection and tracking algorithms. Nevertheless, these approaches still rely on supervised learning schemas with prohibitive annotation demands. To this end, we present CL-MOT, a semi-supervised contrastive learning framework for multi-object tracking. By clustering object embeddings from different views of static frames, CL-MOT learns to discriminate between objects in a scene without relying on identity labels. This simple, yet profound pre-text learning paradigm enables one to transform any object detector into a tracker with no additional annotation labor. We evaluate the proposed approach on the MOT Challenge benchmark, finding that CL-MOT performs on par with supervised trackers and significantly outperforms other self-supervised methods. Interestingly, we even set a new state-of-the-art in re-identification metrics such as IDF1 score, suggesting that CL-MOT learns a more effective feature space than its counterparts. These results demonstrate that our within-frame contrastive learning framework can significantly reduce the annotation demands of tracking while recovering if not surpassing the performance of previous approaches.

## Features
- CL-MOT enables one to train an appearance-based tracker on static images with only bounding box labels
- We do **not** require object identity or association labels, which significantly reducing the annotation demands of tracking
- This repo leverages the one-shot encoder-decoder tracking network from [FairMOT!](https://github.com/ifzhang/FairMOT)
- Our results match the performance of supervised trackers and significantly outperforms other self-supervised methods

## MOT Challenge Results 
Our results are gathered from the  [MOT Challenge!](https://motchallenge.net/) online test server 
| Dataset | MOTA | MOTP | IDF1 | IDSW | MT | ML |
|---------|------|------|------|------|----|----|
| MOT15   |      |      |      |      |    |    |
| MOT17   |      |      |      |      |    |    |
| MOT20   |      |      |      |      |    |    |

### Qualit


## Usage

### Installation

### Dataset Setup

### Training

### Testing

## Acknowledgement

