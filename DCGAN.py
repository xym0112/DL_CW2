class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # First trial - using similar code from VAE and changed the activation function to tanh
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_vector_size, out_channels=num_gen_filters * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_gen_filters * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=num_gen_filters * 4, out_channels=num_gen_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=num_gen_filters * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=num_gen_filters * 2, out_channels=num_gen_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=num_gen_filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=num_gen_filters, out_channels=input_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 


    def forward(self, z, label=None):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # out = self.decoder(z.view(-1, 8, 4, 4))
        out = self.decoder(z)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=num_dis_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=num_dis_filters, out_channels=num_dis_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_dis_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=num_dis_filters * 2, out_channels=num_dis_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_dis_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Conv2d(in_channels=num_dis_filters * 4, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
        
    def forward(self, x, label=None):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        out = self.encoder(x) # B x num_filter * 4 x 4 x 4
        out = self.linear(out).squeeze()
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
        
        return out
