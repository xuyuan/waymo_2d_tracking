Validation

| job   | model                          | Vehicle       | Pedestrian    | Cyclist       | All L1/L2     | LB |
| ----- | ------------------------------ | ------------- | ------------- | ------------- | ------------- | -- |
| 12487 | 12442_epoch10_best_metric      | 0.6654/0.5574 | 0.7075/0.6575 | 0.5624/0.4760 | 0.6451/0.5636 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590081102700001&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12528 | - hflip                        | 0.6657/0.5579 | 0.7073/0.6571 | 0.5630/0.4768 | 0.6453/0.5639 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590143086261058&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12545 | - x1.2                         | 0.6757/0.5697 | 0.7340/0.6863 | 0.5912/0.5030 | 0.6670/0.5864 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590227558716156&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12576 | - x1.4                         | 0.6794/0.5751 | 0.7587/0.7134 | 0.6080/0.5182 | 0.6820/0.6022 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590280949834681&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12608 | 12442_epoch20_best_metric      | 0.6675/0.5599 | 0.7122/0.6621 | 0.5643/0.4761 | 0.6480/0.5660 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590431372302841&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12586 | - x1.6                         | 0.6744/0.5718 | 0.7661/0.7221 | 0.6121/0.5224 | 0.6842/0.6055 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590387036156117&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12559 | WF(12487, 12528)               | 0.6500/0.5371 | 0.7104/0.6605 | 0.5687/0.4845 | 0.6430/0.5607 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590148939244867&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12583 | WF(12489, 12528, 12545, 12576) | 0.6532/0.5440 | 0.7461/0.6997 | 0.5993/0.5135 | 0.6662/0.5857 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590282730178002&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12587 | WF(12545, 12576)               | 0.6574/0.5499 | 0.7505/0.7047 | 0.6049/0.5164 | 0.6710/0.5903 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590308424307307&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12593 | WF[0.6](12545, 12576)          | 0.6602/0.5528 | 0.7527/0.7070 | 0.6065/0.5180 | 0.6731/0.5926 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590335585916754&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12595 | WF[0.7](12545, 12576)          | 0.6623/0.5552 | 0.7531/0.7076 | 0.6067/0.5182 | 0.6740/0.5937 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590338177598942&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12598 | WF[0.8](12545, 12576)          | 0.6653/0.5588 | 0.7518/0.7063 | 0.6059/0.5196 | 0.6743/0.5949 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590348602465373&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12589 | NMS(12545, 12576)              | 0.6797/0.5754 | 0.7574/0.7123 | 0.6096/0.5238 | 0.6822/0.6038 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590320538076129&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12591 | SoftNMS(12545, 12576)          | 0.6925/0.5899 | 0.7584/0.7178 | 0.6083/0.5215 | 0.6864/0.6097 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590328531148147&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12604 | NMS*[0.6](12545, 12576)        | 0.6861/0.5815 | 0.7594/0.7140 | 0.6101/0.5238 | 0.6852/0.6064 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590362961858671&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12611 | SoftNMS*(12545, 12576)         | 0.6798/0.5755 | 0.7555/0.7103 | 0.6095/0.5238 | 0.6816/0.6032 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590409538573172&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12616 | NMW*(12545, 12576)             | 0.6742/0.5715 | 0.7571/0.7120 | 0.6098/0.5238 | 0.6804/0.6024 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590412446495546&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12618 | WF*(12545, 12576)              | 0.6805/0.5755 | 0.7578/0.7124 | 0.6096/0.5209 | 0.6826/0.6029 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590420007342498&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12619 | SoftNMS(12545, 12576, 12528, 12586) | 0.7010/0.6010 | 0.7718/0.7282 | 0.6255/0.5369 | 0.6995/0.6220 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590432925675565&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12619 | 0.02 threshold                      | 0.7009/0.6011 | 0.7719/0.7286 | 0.6257/0.5373 | 0.6995/0.6223 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590438262078955&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |
| 12623 | SoftNMS(all)                   | 0.7000/0.6000 | 0.7703/0.7266 | 0.6189/0.5304 | 0.6964/0.6190 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590457470177105&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&submissionType=VALIDATION) |

Note:
- ensemble method with * means `ensemble_b`
- number means IoU threshold used

Testing Set

| job   | model             | tta  | ensemble |
| ----- | ----------------- | ---- | -------- |
| 12615 | 12442/best_metric | orig | a        |
| 12627 |                   | x1.4 | a        |
| 12628 |                   | x1.2 | a        |
| 12643 |                   | x1.1 ac |  a    |
| 12665 |                   | x1.5,hflip ac |b|
| 12635 | 12620/best_metric | orig | a        |
| 12637 |                   | x1.3 | a        |
| 12644 |                   | x1.6 | a        |
| 12674 | 12650/best_loss   | x1.9,hflip |   c|
| 12670 |                   | x1.6 | b        |
| 12683 |                   | x2   |         c|
| 12705 |                   | x1.4,hflip |   c|
| 12714 |                   | x1.5,hflip,ac |c|

| job   | model              | Vehicle       | Pedestrian    | Cyclist       | All L1/L2     | LB |
| ----- | ------------------ | ------------- | ------------- | ------------- | ------------- | -- |
| 12664 | SoftNMS(a)         | 0.7394/0.6456 | 0.7796/0.7513 | 0.6040/0.5387 | 0.7077/0.6452 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590621467211266&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com) |
| 12687 | SoftNMS(a,b)       | 0.7446/0.6507 | 0.7846/0.7564 | 0.6185/0.5524 | 0.7159/0.6532 | [link](https://waymo.com/open/challenges/entry/?timestamp=1590862328435448&challenge=DETECTION_2D&email=xuyuan.cn@gmail.com) |
| 12718 | SoftNMS(a,b,c)     | 0.7483/0.6583 | 0.7986/0.7694 | 0.6321/0.5664 | 0.7264/0.6647 | [link](https://waymo.com/open/challenges/entry/?challenge=DETECTION_2D&email=xuyuan.cn@gmail.com&timestamp=1590967943540416) |