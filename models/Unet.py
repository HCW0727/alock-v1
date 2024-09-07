import torch
import torch.nn as nn

class Unet2(nn.Module):
    '''
    2-Stage로 구성된 간단한 UNet 모델입니다.
    '''
    def __init__(self, input_tensor):
        super().__init__()

        # Encoder
        self.encoder_cnn1 = nn.Conv2d(in_channels=input_tensor,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        
        self.encoder_cnn2 = nn.Conv2d(in_channels=64,
                                      out_channels=128,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        
        self.encoder_cnn3 = nn.Conv2d(in_channels=128,
                                      out_channels=256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU()
        
        # Decoder
        self.decoder_conv_trans1 = nn.ConvTranspose2d(in_channels=256,
                                               out_channels=128,
                                               kernel_size=2,
                                               stride=2)
        
        self.decoder_cnn1 = nn.Conv2d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        
        self.decoder_conv_trans2 = nn.ConvTranspose2d(in_channels=128,
                                                      out_channels=64,
                                                      kernel_size=2,
                                                      stride=2)
        
        self.decoder_cnn2 = nn.Conv2d(in_channels=128,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.final_fc = nn.Linear(10, 10)

    def forward(self, input_tensor):
        # Encoder
        # Block1
        enc_block1_cnn = self.encoder_cnn1(input_tensor)
        enc_block1_relu = self.relu(enc_block1_cnn)
        enc_block1_maxpool = self.maxpool(enc_block1_relu)

        # Block2
        enc_block2_cnn = self.encoder_cnn2(enc_block1_maxpool)
        enc_block2_relu = self.relu(enc_block2_cnn)
        enc_block2_maxpool = self.maxpool(enc_block2_relu)

        # Block3
        enc_block3_cnn = self.encoder_cnn3(enc_block2_maxpool)
        enc_block3_relu = self.relu(enc_block3_cnn)

        # Decoder
        # enc_Block3
        dec_block3_convT = self.decoder_conv_trans1(enc_block3_relu)
        dec_block3_concat = torch.cat((dec_block3_convT,enc_block2_relu),dim=1)
        dec_block3_conv = self.decoder_cnn1(dec_block3_concat)
        dec_block3_relu = self.relu(dec_block3_conv)

        # Block2
        dec_block2_convT = self.decoder_conv_trans2(dec_block3_relu)
        dec_block2_concat = torch.cat((dec_block2_convT,enc_block1_relu),dim=1)
        dec_block2_conv = self.decoder_cnn2(dec_block2_concat)
        dec_block2_relu = self.relu(dec_block2_conv)

        # Block 2
        out_conv = self.final_conv(dec_block2_relu)
        out_pool = self.adaptive_pool(out_conv)
        out_flatten = self.flatten(out_pool)
        out_fc = self.final_fc(out_flatten)
        return out_fc