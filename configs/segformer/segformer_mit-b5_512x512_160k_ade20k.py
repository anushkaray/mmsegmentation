_base_ = ['./segformer_mit-b0_512x512_160k_ade20k.py']

# model settings
# model = dict(
#     pretrained='pretrain/mit_b5.pth',
#     backbone=dict(
#         embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
#     decode_head=dict(in_channels=[64, 128, 320, 512]))


# model settings
model = dict(
    pretrained='checkpoint_segformer/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
