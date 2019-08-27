import torch
from torch.optim.optimizer import Optimizer, required
import copy


class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super(SGD, self).__init__(params, defaults)
        self.old_params = None
        self.old_grad = None
        self.xminusx = 0
        self.gradminusgrad = 0
        self.grad_norm = 0


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()


        #########START STATS

        #find gradient and params
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
        if self.old_grad is None:
            self.xminusx = params
            self.gradminusgrad = grad
        else:
            self.xminusx = params-self.old_params
            self.gradminusgrad = grad - self.old_grad
        self.grad_norm = torch.norm(grad)
        self.old_params = params
        self.old_grad = grad

        ########END STATS

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


class OnlyStabOleg(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super(OnlyStabOleg, self).__init__(params, defaults)
        self.params_k=None
        self.xminusx = 0
        self.grad_norm = 0


    def step(self, closure=None):
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



class BBStabOleg(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super(BBStabOleg, self).__init__(params, defaults)
        self.old_params = None
        self.old_grad = None
        self.xminusx = 0
        self.gradminusgrad = 0
        self.grad_norm = 0


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            #find gradient and params
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
            if self.old_grad is None:
                self.xminusx = params
                self.gradminusgrad = grad
            else:
                self.xminusx = params-self.old_params
                self.gradminusgrad = grad - self.old_grad
            self.grad_norm = torch.norm(grad)

            #calculate alpha
            alpha_stab = 1/self.grad_norm
            alpha_bb = torch.sum(self.xminusx*self.gradminusgrad)/torch.sum(self.gradminusgrad*self.gradminusgrad)
            if self.old_grad is None:
                alpha = alpha_stab
            else:
                alpha = min(alpha_stab, alpha_bb)

            #do update
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    p.data.add_(-group['lr']*alpha, d_p)
            
            self.old_params = params
            self.old_grad = grad



        return loss
