# Mesh to PC to Voxelized PC conversion

Example usage:

    python mesh_to_vpc.py --vg_size 10 --n_samples 32000000 test.obj test.ply

Split in two steps:

    python mesh_to_pc.py --n_samples 32000000 test.obj test_vox10.ply
    python pc_to_vpc.py --vg_size 10 test.obj test_vox10.ply

# G-PCC running

    python ~/code/2021_pc_perceptual_dataset/src/run_gpcc.py \                 
            ./pc_quality_dataset_vpc "*.ply" \
            ./pc_quality_dataset_vpc_ppc/gpcc-oct-predlift/ \
            ~/code/MPEG/mpeg-pcc-tmc13-v14.0/ \
            ~/code/MPEG/mpeg-pcc-tmc13-v14.0/cfg/octree-predlift/lossy-geom-lossy-attrs/longdress_vox10_1300/ \
            ~/code/MPEG/mpeg-pcc-dmetric-v0.13.5/test/pc_error

# pcc_geo_cnn_v2 running

    for pc in ~/datassd/pc_quality_dataset_vpc/*; do for i in 1.00e-05 2.00e-05 3.00e-04 5.00e-05; do echo python ev_experiment.py \
            --output_dir ~/datassd/pc_quality_dataset_vpc_ppc/pcc_geo_cnn_v2/${$(basename $pc)%%.ply}/${i} \
            --model_dir ~/datassd/experiments/c4-ws/${i} \
            --model_config c3p --opt_metrics d1_mse d2_mse --max_deltas inf \
            --pc_name ${pc%%*/} \
            --pcerror_path ~/code/MPEG/mpeg-pcc-dmetric-v0.13.5/test/pc_error \
            --pcerror_cfg_path ~/code/MPEG/mpeg-pcc-tmc13-v14.0/cfg/octree-predlift/lossy-geom-lossy-attrs/longdress_vox10_1300/r06/pcerror.cfg \
            --input_pc $pc \
            --input_norm $pc; done; done

# Render image

    python render_img.py 001.ply 001.png

# Render video

    python render_vid.py 001.ply 001.avi
