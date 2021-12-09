python age_lsgan_transfer.py \
  --gan_loss_weight=75 \
  --fea_loss_weight=0.5e-4 \
  --age_loss_weight=30 \
  --fea_layer_name=conv5 \
  --checkpoint_dir=./checkpoints/age/0_conv5_lsgan_transfer_g75_0.5f-4_a30 \
  --sample_dir=age/0_conv5_lsgan_transfer_g75_0.5f-4_a30 
