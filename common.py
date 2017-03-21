

__all__ = (
    'LETTERS',
    'CHARS',
    'sigmoid',
    'softmax',
)

import numpy

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHARS = LETTERS

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
  return 1. / (1. + numpy.exp(-a))

