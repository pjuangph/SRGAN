import torch
from torch import nn
from torchvision.models.vgg import vgg16

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        '''
            Args
                out_labels - this is if the generated image (out_image) matches the target image
                out_image - generated image G_thetaG(I^{LR})
                target_image - Reference image I^{HR}
        '''
        # Adversarial Loss - Right part of equation 3
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss - this is equation 5
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images)) # VGG16
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images) # Content loss euqation 3 (left part)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        #  Equation 3: Content_Loss + 1E-3 * Adversarial_loss +(VGG19 Loss)+ (Loss super resolution VGG/2.2)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss




if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
