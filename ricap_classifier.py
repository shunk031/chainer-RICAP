import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import reporter
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy


class RICAPClassifier(L.Classifier):

    NUM_PATCH = 4

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 label_key=-1,
                 beta=0.3):
        super(RICAPClassifier, self).__init__(
            predictor, lossfun=lossfun,
            accfun=accfun, label_key=label_key)

        self.beta = beta

    def RICAP(self, inputs, target):

        I_x, I_y = inputs.shape[2:]
        w = int(np.round(I_x * np.random.beta(self.beta, self.beta)))
        h = int(np.round(I_y * np.random.beta(self.beta, self.beta)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        cropped_images = []
        c_ = []
        W_ = []
        for k in range(self.NUM_PATCH):
            idx = np.random.permutation(inputs.shape[0])
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images.append(inputs[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]])
            c_.append(target[idx])
            W_.append(w_[k] * h_[k] / (I_x * I_y))

        patched_images = F.concat((
            F.concat((cropped_images[0], cropped_images[1]), axis=2),
            F.concat((cropped_images[2], cropped_images[3]), axis=2)), axis=3)

        return patched_images, c_, W_

    def forward(self, *args, **kwargs):
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None

        if chainer.config.train:
            patched_images, c, W = self.RICAP(*args, t)
            self.y = self.predictor(patched_images)
            self.loss = sum([W[k] * self.lossfun(self.y, c[k]) for k in range(self.NUM_PATCH)])
        else:
            self.y = self.predictor(*args, **kwargs)
            self.loss = self.lossfun(self.y, t)

        reporter.report({'loss': self.loss}, self)

        if self.compute_accuracy:
            if chainer.config.train:
                self.accuracy = sum([W[k] * F.accuracy(self.y, c[k]) for k in range(self.NUM_PATCH)])
            else:
                self.accuracy = self.accfun(self.y, t)

            reporter.report({'accuracy': self.accuracy}, self)

        return self.loss
