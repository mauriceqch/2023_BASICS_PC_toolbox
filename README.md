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


# Adaptive rendering

    positionQuantizationScale: 0.125
    positionQuantizationScale: 0.25
    positionQuantizationScale: 0.5
    positionQuantizationScale: 0.75
    positionQuantizationScale: 0.875
    positionQuantizationScale: 0.9375

    python render_img.py 001_GPCC_r01.ply 001_GPCC_r01_c1.png --cube_size 1
    python render_img.py 001_GPCC_r02.ply 001_GPCC_r02_c1.png --cube_size 1
    python render_img.py 001_GPCC_r03.ply 001_GPCC_r03_c1.png --cube_size 1
    python render_img.py 001_GPCC_r04.ply 001_GPCC_r04_c1.png --cube_size 1
    python render_img.py 001_GPCC_r05.ply 001_GPCC_r05_c1.png --cube_size 1
    python render_img.py 001_GPCC_r06.ply 001_GPCC_r06_c1.png --cube_size 1

    python render_img.py 001_GPCC_r01.ply 001_GPCC_r01_ca.png --cube_size 8
    python render_img.py 001_GPCC_r02.ply 001_GPCC_r02_ca.png --cube_size 4
    python render_img.py 001_GPCC_r03.ply 001_GPCC_r03_ca.png --cube_size 2
    python render_img.py 001_GPCC_r04.ply 001_GPCC_r04_ca.png --cube_size 1
    python render_img.py 001_GPCC_r05.ply 001_GPCC_r05_ca.png --cube_size 1
    python render_img.py 001_GPCC_r06.ply 001_GPCC_r06_ca.png --cube_size 1
