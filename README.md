# Advanced Machine Learning Project - ROS

## Txt_List

In new_txt_list you can find some mockup data to test step two

## Weights

Too heavy for github: https://mega.nz/folder/Dd50HDqb#RTR0tPVztY23TdVBff_hLA

## Art -> Clipart

_Epochs 40_

**Aucroc**

|     |  1  |   3    |   10   |   20   |
| :-: | :-: | :----: | :----: | :----: |
| 20  |     | 0.4819 | 0.4709 |        |
| 40  |     | 0.4687 | 0.4728 | 0.4684 |
| 80  |     |        | 0.4799 |        |
| 120 |     |        | 0.4941 |        |

**Threshold**

Must: 3064

|      | Known |
| :--: | :---: |
| 0.3  | 2927  |
| 0.5  | 1761  |
| 0.75 |  739  |

**Step2**

WR2 = 10

|     |  OS\*  |  UNK   |  HOS   |
| :-: | :----: | :----: | :----: |
| 10  | 0.0530 | 0.9076 | 0.1002 |
| 20  | 0.0733 | 0.8603 | 0.1352 |

WR2 = 0.1

|     |  OS\*  |  UNK   |  HOS   |
| :-: | :----: | :----: | :----: |
| 10  | 0.1051 | 0.8273 | 0.1864 |
| 20  | 0.1001 | 0.8163 | 0.1783 |

## Product -> Art

_Epochs 40_

Aucroc WR 10: 0.4956

**Threshold**

Must: 1789

|     | Known |
| :-: | :---: |
| 0.1 | 1964  |
| 0.3 |  12   |
| 0.5 |   0   |

**Step2**

WR2 = 0.1

|     |  OS\*  |  UNK   |  HOS   |
| :-: | :----: | :----: | :----: |
| 10  | 0.2679 | 0.4991 | 0.3487 |
| 20  | 0.2874 | 0.4877 | 0.3617 |

## RealWorld -> Product

_Epochs 40_

Aucroc WR 10: 0.5341

**THreshold**

Must: 3143

|     | Known |
| :-: | :---: |
| 0.3 | 3551  |
| 0.5 | 2641  |

**Step2**

WR2 = 0.1

|     |  OS\*  |  UNK   |  HOS   |
| :-: | :----: | :----: | :----: |
| 10  | 0.5067 | 0.4830 | 0.4946 |
| 20  | 0.5155 | 0.4374 | 0.4733 |

## DOCS

https://github.com/silvia1993/ROS
https://github.com/gidariss/FeatureLearningRotNet
