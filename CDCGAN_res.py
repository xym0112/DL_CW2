# *CODE FOR PART 2.1 IN THIS CELL*
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.fc = nn.Sequential(
            nn.Linear(latent_vector_size + num_CIFAR_label, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (input_size // 4) * (input_size // 4)),
            nn.BatchNorm1d(128 * (input_size // 4) * (input_size // 4)),
            nn.ReLU(),
        )

        self.deconv1 = nn.Sequential(
            GenResBlock(128, 128),
            GenResBlock(128, 128),
            GenResBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channel, 4, 2, 1),
            nn.Tanh(),
        )

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 


    def forward(self, z, label):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        label_g = F.one_hot(label, num_CIFAR_label).float().to(device)
        z = torch.cat([z, label_g], 1)
        z = self.fc(z)
        z = z.view(-1, 128, (input_size // 4), (input_size // 4))
        out = self.deconv1(z)
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

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel + num_CIFAR_label, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (input_size // 4) * (input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
        
    def forward(self, x, label):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        label_d = torch.zeros((len(x), num_CIFAR_label)).scatter_(1, label.type(torch.LongTensor).unsqueeze(1), 1)
        label_d = label_d.unsqueeze(2).unsqueeze(3).expand(len(x), num_CIFAR_label, input_size, input_size).to(device)
        x = torch.cat([x, label_d], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (input_size // 4) * (input_size // 4))
        out = self.fc(x).squeeze()
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
        
        return out


class GenResBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(GenResBlock, self).__init__()
        #311
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_feat, in_feat, 3, 1, 1),
            nn.BatchNorm2d(in_feat),
            nn.ReLU(),
            nn.ConvTranspose2d(in_feat, out_feat, 3, 1, 1),
            nn.BatchNorm2d(out_feat))

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.layers(x)
        # print('res block shape: {}'.format(out.shape))
        out += residual
        
        return out
        
train_losses_G = []
train_losses_D = []

for epoch in range(num_epochs):
    train_loss_D = 0
    train_loss_G = 0
    for i, (image, label) in enumerate(loader_train, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################device
        # train with real
        optimizerD.zero_grad()

        for p in model_G.parameters():
          p.requires_grad = False
        
        for p in model_D.parameters():
          p.requires_grad = True

        real_cpu = image.to(device)
        label = label.to(device)
        real_labels = torch.full((len(real_cpu),), fill_value=real_label, dtype=torch.float, device=device)

        output = model_D(real_cpu, label=label)
        errD_real = loss_function(output, real_labels)
        D_x = output.mean().item()

        # train with fake

        noise = torch.randn(len(real_cpu), latent_vector_size, device=device)
        with torch.no_grad():
          fake = model_G(noise, label=label)
          # print(fake.shape)
        fake_labels = torch.full((len(real_cpu),), fill_value=fake_label, dtype=torch.float, device=device)

        output = model_D(fake, label)
        errD_fake = loss_function(output, fake_labels)
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        errD.backward()
        train_loss_D += errD.item()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()

        for p in model_G.parameters():
          p.requires_grad = True
        
        for p in model_D.parameters():
          p.requires_grad = False

        noise = torch.randn(len(real_cpu), latent_vector_size, device=device)
        real_labels = torch.full((len(real_cpu),), fill_value=real_label, dtype=torch.float, device=device)

        fake = model_G(noise, label)
        output = model_D(fake, label)
        D_G_z2 = output.mean().item()

        errG = loss_function(output, real_labels)
        errG.backward()
        train_loss_G += errG.item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(loader_train),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    if epoch == 0:
        # save_image(denorm(real_cpu.cpu()).float(), '/content/drive/MyDrive/icl_dl_cw2/CW_GAN/real_samples.png')
        save_image(denorm(real_cpu.cpu()).float(), './CW_GAN/real_samples.png')
    with torch.no_grad():
        fake = model_G(fixed_noise, fixed_labels)
        # save_image(denorm(fake.cpu()).float(), '/content/drive/MyDrive/icl_dl_cw2/CW_GAN/fake_samples_epoch_%03d.png' % epoch)
        save_image(denorm(fake.cpu()).float(), './CW_GAN/fake_samples_epoch_%03d.png' % epoch)
    train_losses_D.append(train_loss_D / len(loader_train))
    train_losses_G.append(train_loss_G / len(loader_train))
