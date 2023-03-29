import math
from abc import ABC, abstractmethod


class OneCycleScheduler(ABC):
    def __init__(self, optimizer, lr, warmup_iterations=0, decay_iterations=1, lr_start_coeff=1., lr_final_coeff=1.):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_iterations = warmup_iterations - 1  # -1 because we start from 0
        self.decay_iterations = decay_iterations - 1  # -1 because we start from 0
        self.lr_start_coeff = lr_start_coeff
        self.lr_start = lr * lr_start_coeff
        self.lr_final_coeff = lr_final_coeff
        self.lr_final = lr * lr_final_coeff
        self.iteration = -1

    @abstractmethod
    def warmup_fn(self):
        pass

    @abstractmethod
    def decay_fn(self):
        pass

    def get_lr(self):
        assert self.warmup_iterations < self.decay_iterations, "Warmup iterations must be less than decay iterations"

        if self.iteration <= self.warmup_iterations:
            return self.warmup_fn()
        elif self.iteration > self.decay_iterations:
            return self.lr_final
        else:
            return self.decay_fn()

    def step(self):
        self.iteration += 1
        lr_new = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_new


class CosineDecay(OneCycleScheduler):
    def warmup_fn(self):
        # linear warmup
        warmup_rate = self.iteration / self.warmup_iterations
        lr_new = self.lr_start + warmup_rate * (self.lr - self.lr_start)
        return lr_new
    
    def decay_fn(self):
        # cosine decay - https://arxiv.org/pdf/1812.01187.pdf
        decay_rate = 0.5 * (1.0 + math.cos(((self.iteration - self.warmup_iterations) * math.pi) / (self.decay_iterations - self.warmup_iterations)))
        lr_new = self.lr_final + decay_rate * (self.lr - self.lr_final)
        return lr_new
    

class LinearDecay(OneCycleScheduler):
    def warmup_fn(self):
        # linear warmup
        warmup_rate = self.iteration / self.warmup_iterations
        lr_new = self.lr_start + warmup_rate * (self.lr - self.lr_start)
        return lr_new
    
    def decay_fn(self):
        # linear decay
        decay_rate = 1 - (self.iteration - self.warmup_iterations) / (self.decay_iterations - self.warmup_iterations)
        lr_new = self.lr_final + decay_rate * (self.lr - self.lr_final)
        return lr_new
    

class NoneDecay(OneCycleScheduler):
    def warmup_fn(self):
        return self.lr
    
    def decay_fn(self):
        return self.lr


if __name__ == "__main__":
    scheduler = CosineDecay(None, lr=0.1, warmup_iterations=10, decay_iterations=100, lr_start_coeff=0.2, lr_final_coeff=0.1)

    for i in range(100):
        scheduler.iteration += 1
        print(i+1, scheduler.get_lr())

