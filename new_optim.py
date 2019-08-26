import torch
from torch.optim.optimizer import Optimizer, required
import copy


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss = loss_fn(model(input), target)
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super(SGD, self).__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.data.add_(-group['lr'], d_p)

        return loss


class OlegOptim(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss = loss_fn(model(input), target)
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super(OlegOptim, self).__init__(params, defaults)
        self.params_k=None
        self.xminusx = 0
        self.grad_norm = 0


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            #find gradient
            grad = []
            params = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad += [p.grad.data.view(-1)]
                    params += [p.data.view(-1)]
            grad = torch.cat(grad)
            params = torch.cat(params)

            #calculate delta and alpha
            if self.params_k is None:
                self.xminusx = torch.norm(grad)
            else:
                self.xminusx = torch.norm(params-self.params_k)
            self.grad_norm = torch.norm(grad)
            self.params_k = params

            #do update
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    p.data.add_(-group['lr']/self.grad_norm, d_p)


        return loss
