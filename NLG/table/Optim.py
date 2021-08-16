import torch.optim as optim
from torch.nn.utils import clip_grad_norm


class Optim(object):

    def set_parameters(self, params):
        self.params = [p for p in params if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.params, lr=self.lr, alpha=self.alpha)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr,
                                        betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, alpha, max_grad_norm,
                 lr_decay=1, start_decay_at=None,
                 beta1=0.9, beta2=0.98,
                 opt=None):
        self.last_metric = None
        self.lr = lr
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.opt = opt

    def _setRate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        "Compute gradients norm."
        self._step += 1

        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def updateLearningRate(self, metric, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if (self.start_decay_at is not None) and (epoch >= self.start_decay_at):
            self.start_decay = True
        if (self.last_metric is not None) and (metric is not None) and (metric > self.last_metric):
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_metric = metric
        self.optimizer.param_groups[0]['lr'] = self.lr
