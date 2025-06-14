import torch
from torch import nn
from base import BaseModel
from models.transformer.encoder_layer import Postnet
from models.transformer.encoder import ConformerEncoder


class Speech2Landmark(BaseModel):
    def __init__(self, args):
        super().__init__()
        
        self.encoder = ConformerEncoder(input_layer="conv2d",
            input_size=args.in_dim, output_size=256, num_blocks=args.num_encoder_block)

        self.decoder = ConformerEncoder(input_layer="linear", 
            input_size=256, output_size=256, num_blocks=args.num_decoder_block)
        
        self.out_proj = nn.Linear(256, args.out_dim)

        self.pos_net = Postnet(embed_dim=128,
            out_dim=args.out_dim, kernel_size=5, n_conv=args.num_postnet_cnn_layer)
        
        self.test_layer = nn.Linear(1024, 256)
    
    def forward(self, x, x_len):
        """
        Args:
            x (torch.Tensor): (batch_size, seq_len, input_dim)
            x_len (torch.Tensor): (batch_size, )
        Returns:
            predict_mid (torch.Tensor): (batch_size, seq_len//2, output_dim)
            predict (torch.Tensor): (batch_size, seq_len//2, output_dim)
        """
        x, mask = self.encoder(x, xs_lens=x_len)
        x, _ = self.decoder(x, masks=mask)
        predict_mid = self.out_proj(x)
        predict = self.pos_net(predict_mid.transpose(1, 2)).transpose(1, 2)
        
        return predict_mid, predict
    
