from torch import cat
from torch.nn import Module, Sequential, Linear, LSTM, Embedding, Softmax
from torchvision.models import vgg16_bn
from constants import *

class Encoder(Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        model = vgg16_bn(pretrained=True)
        self.model = Sequential(*(list(model.children())[:-1]))
        self.embed = Linear(list(model.classifier.children())[0].in_features, out_features=embed_size)
        # print(self.model)
    
    def forward(self, image):
        print("\t\tExtracting features")
        features = self.model(image)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        print("\t\tFeatures extracted")
        return features


class Decoder(Module):
    def __init__(self, lstm_units, embed_size, num_layers, vocab_size):
        super(Decoder, self).__init__()
        self.lstm_units = lstm_units
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed = Embedding(self.vocab_size, self.embed_size)
        self.lstm = LSTM(self.embed_size, self.lstm_units, self.num_layers, batch_first=True, bias=True)
        self.linear = Linear(self.lstm_units, vocab_size)
        self.sf = Softmax(dim=1)
    
    def forward(self, caption, image_features, hidden):
        print("\t\tRunning LSTM")
        caption = caption[:,:-1]
        embed = self.embed(caption)
        inputs = cat((image_features.unsqueeze(1), embed), 1)
        outputs, hidden = self.lstm(inputs, hidden)
        logits = self.linear(outputs)
        logits = self.sf(logits)
        print("\t\tLSTM Done.")
        return logits, hidden
    
    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        return (weights.new(self.num_layers, batch_size, self.lstm_units).zero_(), 
                weights.new(self.num_layers, batch_size, self.lstm_units).zero_())
        




