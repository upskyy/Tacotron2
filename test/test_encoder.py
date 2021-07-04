import unittest
import torch
from tacotron2.encoder import Encoder


class TestTacotron2Encoder(unittest.TestCase):
    def test_forward(self):
        num_vocabs = 10

        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        encoder = Encoder(num_vocabs).to(device)
        print(encoder)

        inputs = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                    [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                    [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
        input_lengths = torch.LongTensor([9, 8, 7]).to(device)

        output = encoder(inputs, input_lengths)
        print(output)
        print(output.size())