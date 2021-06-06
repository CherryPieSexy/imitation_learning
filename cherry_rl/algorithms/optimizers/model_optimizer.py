import torch


class ModelOptimizer:
    """
    Optimizer for any model (actor-critic, discriminator, dynamics model).
    Encoder is part of actor-critic and optimized inside.
    """
    def __init__(
            self,
            model,
            learning_rate, clip_grad,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        self.clip_grad = clip_grad

    @staticmethod
    def _average_loss(loss, mask):
        if mask is None:
            mask = torch.ones_like(loss)
        return (mask * loss).sum() / mask.sum()

    def optimize_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return grad_norm

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def train(self, *args):
        raise NotImplementedError