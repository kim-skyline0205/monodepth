import random
# # a는 이어서 쓰기 w는 새로 쓰기

######### cityscape extra
#A=random.sample(range(200),200)
fw = open('kitti_stereo_test_rignt_image_list.txt','w')
for r in range(200):
    i=str(r).zfill(4)
    fw.write('/media/keti/BAE6-BC42/DataSets/KITTI/Stereo_Evaluation_2015/data_scene_flow/training/image_3/00'+i+'_10.jpg''\n')
fw.close()
#
#
#A=random.sample(range(200),200)
fw = open('kitti_stereo_test_left_disparity_list.txt','a')
for r in range(200):
   # i = str(A[r]).zfill(4)
    i = str(r)
    fw.write('/home/keti/Desktop/monodepth/result_generation/cityscapes_kitti_resnet50/kitti_2015_stereo/PP_left_result/'+i+'.png''\n')
fw.close()