import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained

class FBModel(object):
    def __init__(self, name, repr_layer=[-1]):
        self.name_ = name
        self.repr_layer_ = repr_layer

        model, alphabet = pretrained.load_model_and_alphabet(name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self.model_ = model
        self.alphabet_ = alphabet

        assert(all(
            -(model.num_layers + 1) <= i <= model.num_layers
            for i in [ -1 ]
        ))
        self.repr_layers_ = [
            (i + model.num_layers + 1) % (model.num_layers + 1)
            for i in [ -1 ]
        ]
