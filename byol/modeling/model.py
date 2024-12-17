import torch.nn as nn
from torch import no_grad


class Encoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class Projector(nn.Module):
    def __init__(self, projector):
        super().__init__()
        self.projector = projector

    def forward(self, x):
        return self.projector(x)


class Predictor(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, x):
        return self.predictor(x)


class BootstrapYourOwnLatent(nn.Module):
    def __init__(self, encoder, projector, predictor, tau):
        super().__init__()

        self.tau = tau

        self.online_encoder = encoder
        self.online_projector = projector
        self.online_predictor = predictor

        self.target_encoder = encoder
        self.target_projector = projector

    def update_the_moving_average_for_the_encoder(self):
        with no_grad():
            for theta, ksi in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
            ):
                new_ksi = self.tau * ksi + (1 - self.tau) * theta
                ksi.data = new_ksi

    def update_the_moving_average_for_the_projector(self):
        with no_grad():
            for theta, ksi in zip(
                self.online_projector.parameters(), self.target_projector.parameters()
            ):
                new_ksi = self.tau * ksi + (1 - self.tau) * theta
                ksi.data = new_ksi

    @staticmethod
    def compute_loss(x, y):
        x = nn.functional.normalize(x, dim=-1)
        y = nn.functional.normalize(y, dim=-1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, v1, v2):
        online_prediction1 = self.online_predictor(self.online_projector(self.online_encoder(v1)))
        online_prediction2 = self.online_predictor(self.online_projector(self.online_encoder(v2)))

        with no_grad():
            target_prediction1 = self.target_projector(self.target_encoder(v1))
            target_prediction2 = self.target_projector(self.target_encoder(v2))

        loss = (
            self.compute_loss(online_prediction1, target_prediction1).mean()
            + self.compute_loss(online_prediction2, target_prediction2).mean()
        )
        return loss


class FineTunedBootstrapYourOwnLatent(nn.Module):

    def __init__(self, encoder, fine_tuning_mlp):
        super().__init__()
        self.mlp = fine_tuning_mlp
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x
