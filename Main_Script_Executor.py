import os

Schedule = []                

# ========= EXPERIMENTS TRAINING WITH in the same image (MG_10m) ============
# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase train \
#                                 --batch_size 1 --epoch 60 --dataset_name MG_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128  \
#                                 --patch_overlap 0.0 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase generate_complete_image \
#                                 --batch_size 1 --epoch 60 --dataset_name MG_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128  \
#                                 --patch_overlap 0.0 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator deeplab --discriminator atrous --phase train \
#                                 --batch_size 2 --epoch 60 --dataset_name MG_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 256 --output_stride 16 \
#                                 --patch_overlap 0.35 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator deeplab --discriminator atrous --phase generate_complete_image \
#                                 --batch_size 2 --epoch 60 --dataset_name MG_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 256 --output_stride 16 \
#                                 --patch_overlap 0.35 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --phase GEE_metrics --dataset_name MG_10m --test_dir ./test")
# Schedule.append("python main.py --phase Meraner_metrics --dataset_name MG_10m --test_dir ./test")








# ========= EXPERIMENTS TRAINING WITH in the same image (Para_10m) ============
# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase train \
#                                 --batch_size 1 --epoch 60 --dataset_name Para_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128  \
#                                 --patch_overlap 0.0 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase generate_complete_image \
#                                 --batch_size 1 --epoch 60 --dataset_name Para_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128  \
#                                 --patch_overlap 0.0 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

'''
Schedule.append("python main.py --generator deeplab --discriminator atrous --phase generate_complete_image \
                                --batch_size 2 --epoch 60 --dataset_name Santarem_I5 \
                                --datasets_dir E:/Jorge/dataset/ --image_size_tr 256 --output_stride 16 \
                                --patch_overlap 0.4 --date both\
                                --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")
    Schedule.append("python main.py --generator deeplab --discriminator atrous --phase generate_complete_image \
                                    --batch_size 2 --epoch 60 --dataset_name Santarem \
                                    --datasets_dir E:/Jorge/dataset/ --image_size_tr 256 --output_stride 16 \
                                    --patch_overlap 0.4 --date both\
                                    --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")
'''
mode = 'infer'

if mode == 'train':
#    Schedule.append("python main.py --generator deeplab --discriminator atrous --phase train \
#                                    --batch_size 2 --epoch 60 --dataset_name MG_10m \
#                                    --datasets_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/ --image_size_tr 256 --output_stride 16 \
#                                    --patch_overlap 0.4 --date both\
#                                    --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

    Schedule.append("python main.py --generator deeplab --discriminator atrous --phase train \
                                    --batch_size 2 --epoch 60 --dataset_name Para_10m \
                                    --datasets_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/ --image_size_tr 256 --output_stride 16 \
                                    --patch_overlap 0.4 --date both\
                                    --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test\
                                    ")

#                                     --SSIM_lambda 100.0")

elif mode == 'infer':

    # Schedule.append("python main.py --generator deeplab --discriminator pix2pix --phase generate_complete_image \
    #                                 --batch_size 2 --epoch 60 --dataset_name NRW \
    #                                 --datasets_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/ --image_size_tr 256 --output_stride 16 \
    #                                 --patch_overlap 0.4 --date d0\
    #                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")
    Schedule.append("python main.py --generator deeplab --discriminator atrous --phase generate_complete_image \
                                    --batch_size 2 --epoch 60 --dataset_name MG_10m \
                                    --datasets_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/ --image_size_tr 256 --output_stride 16 \
                                    --patch_overlap 0.4 --date both\
                                    --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")


elif mode == 'GEE_metrics':
    # Schedule.append("python main.py --phase GEE_metrics --dataset_name NRW --test_dir ./test")
    Schedule.append("python main.py --phase GEE_metrics --dataset_name NRW --test_dir E:/Jorge/dataset/ --datasets_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/")

elif mode == 'Meraner_metrics':
    Schedule.append("python main.py --phase Meraner_metrics --dataset_name NRW \
                                    --test_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/NRW \
                                    --datasets_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/")


# ========= EXPERIMENTS TRAINING WITH SEN2MS-CR SCENES ============
# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase train \
#                                 --batch_size 1 --epoch 60 --dataset_name SEN2MS-CR \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128 --output_stride 8 \
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase generate_complete_image \
#                                 --batch_size 1 --dataset_name Para_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128 --output_stride 8 \
#                                 --checkpoint_dir ./checkpoint --test_dir ./test")

# Schedule.append("python main.py --generator deeplab --discriminator atrous --phase train \
#                                 --batch_size 2 --epoch 60 --dataset_name SEN2MS-CR \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 256 --output_stride 16 \
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator deeplab --discriminator atrous --phase generate_complete_image \
#                                 --batch_size 2 --dataset_name Para_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128 --output_stride 8 \
#                                 --checkpoint_dir ./checkpoint --test_dir ./test")



# Schedule.append("python main.py --phase GEE_metrics")

# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase generate_image_patches \
#                                 --dataset_name SEN2MS-CR \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128 --output_stride 8 \
#                                 --checkpoint_dir ./checkpoint --test_dir ./test")



for i in range(len(Schedule)):
    os.system(Schedule[i])
