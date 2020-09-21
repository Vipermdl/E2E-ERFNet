## E2E-ERFNet

The re-implementation of &lt;End-to-End Lane Marker Detection via Row-wise Classification>

The original paper implemented CULane dataset for 74% F1 score. We re-implemented the models for 67.2%. We believe that there are some differences in the implementation details compared with the original text. (We didn't add dropout operation to the network because it dropped 2% F1 score.)

## result

------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test0_normal.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 28271 fp: 4438 fn: 4506
finished process file
precision: 0.864319
recall: 0.862526
Fmeasure: 0.863421
----------------------------------
------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test1_crowd.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 18681 fp: 9001 fn: 9322
finished process file
precision: 0.674843
recall: 0.667107
Fmeasure: 0.670953
----------------------------------
------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test2_hlight.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 963 fp: 647 fn: 722
finished process file
precision: 0.598137
recall: 0.571513
Fmeasure: 0.584522
----------------------------------
------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test3_shadow.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 1756 fp: 1118 fn: 1120
finished process file
precision: 0.610995
recall: 0.61057
Fmeasure: 0.610783
----------------------------------
------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test4_noline.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 5544 fp: 7611 fn: 8477
finished process file
precision: 0.421437
recall: 0.395407
Fmeasure: 0.408007
----------------------------------
------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test5_arrow.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 2492 fp: 592 fn: 690
finished process file
precision: 0.808042
recall: 0.783155
Fmeasure: 0.795404
----------------------------------
------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test6_curve.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 707 fp: 438 fn: 605
finished process file
precision: 0.617467
recall: 0.538872
Fmeasure: 0.575499
----------------------------------
------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test7_cross.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 0 fp: 2841 fn: 0
no ground truth positive
finished process file
precision: 0
recall: -1
Fmeasure: 0
----------------------------------
------------Configuration---------
anno_dir: /mnt/HD/dataset/CULane/
detect_dir: ./result/culane_eval_tmp/
im_dir: /mnt/HD/dataset/CULane/
list_im_file: /mnt/HD/dataset/CULane/list/test_split/test8_night.txt
width_lane: 30
iou_threshold: 0.5
im_width: 1640
im_height: 590
-----------------------------------
Evaluating the results...
tp: 12424 fp: 8545 fn: 8606
finished process file
precision: 0.592494
recall: 0.590775
Fmeasure: 0.591633
----------------------------------
res_normal 0.863421
res_crowd 0.670953
res_night 0.591633
res_noline 0.408007
res_shadow 0.610783
res_arrow 0.795404
res_hlight 0.584522
res_curve 0.575499
res_cross 0.0
0.6715934678011898

## Thanks

Some codes implemented by [Ultra-fast-laneNet](https://github.com/cfzd/Ultra-Fast-Lane-Detection), we appreciated for their works. 
