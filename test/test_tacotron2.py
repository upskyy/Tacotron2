import unittest
import torch
from tacotron2.encoder import Encoder
from tacotron2.decoder import Decoder
from tacotron2 import Tacotron2


class TestTacotron2(unittest.TestCase):
    def test_encoder(self):
        num_vocabs = 10

        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        encoder = Encoder(num_vocabs).to(device)
        print(encoder)

        inputs = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                    [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                    [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
        input_lengths = torch.LongTensor([9, 8, 7]).to(device)

        outputs = encoder(inputs, input_lengths)

        print(outputs)
        print(outputs.size())

    def test_decoder(self):
        batch = 3
        encoder_seq_length = 20
        decoder_seq_length = 100
        encoder_dim = 512
        n_mel = 80

        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        decoder = Decoder().to(device)
        print(decoder)

        encoder_outputs = torch.FloatTensor(batch, encoder_seq_length, encoder_dim).to(device)
        decoder_inputs = torch.FloatTensor(batch, decoder_seq_length, n_mel).to(device)

        outputs = decoder(encoder_outputs, decoder_inputs)

        print(outputs["feature_outputs"].size())
        print(outputs["stop_tokens"].size())
        print(outputs["attention_weights"].size())

    def test_tacotron2_train(self):
        batch = 3
        seq_length = 100
        n_mel = 80
        num_vocabs = 10

        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        model = Tacotron2(num_vocabs=num_vocabs).to(device)
        print(model)

        inputs = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                   [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                   [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
        input_lengths = torch.LongTensor([9, 8, 7]).to(device)
        target = torch.FloatTensor(batch, seq_length, n_mel).to(device)

        outputs = model(inputs, input_lengths, target, teacher_forcing_ratio=1.0)

        print(outputs["feature_outputs"].size())
        print(outputs["stop_tokens"].size())
        print(outputs["attention_weights"].size())

    def test_tacotron2_inference(self):
        num_vocabs = 10

        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        model = Tacotron2(num_vocabs=num_vocabs).to(device)
        print(model)

        inputs = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                   [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                   [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
        input_lengths = torch.LongTensor([9, 8, 7]).to(device)

        outputs = model(inputs, input_lengths, teacher_forcing_ratio=0.0)

        print(outputs["feature_outputs"].size())
        print(outputs["stop_tokens"].size())
        print(outputs["attention_weights"].size())