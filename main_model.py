import numpy as np
import math
import os

from torch import Tensor, reshape, save, unsqueeze, load, device, no_grad
from torch.nn import CrossEntropyLoss
from torch.optim import Adagrad
from constants import *
from custom_dataset import CustomDataset
from helper_models import Encoder, Decoder
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image

class CaptionGenerator():
    def __init__(self, root_dir, captions_file_name, instances_ann_name, image_dir, transform, embed_size=512, num_layers=1, lstm_units=512, mode='train'):
        print("INIT")
        (self.train_loader, self.val_loader) = self.get_data_loader(root_dir, captions_file_name, instances_ann_name, image_dir, mode, transform)
        self.device = device('cuda')
        self.encoder = Encoder(embed_size).to(self.device)
        self.decoder = Decoder(lstm_units, embed_size, num_layers, self.train_loader.dataset.vocab.get_vocab_size()).to(self.device)
        self.loss = []
        self.val_loss = []
        self.bleu_scores = []
        self.blue_best = float("-INF")
        self.num_train_batches = len(self.train_loader)
        self.num_val_batches = len(self.val_loader)
    
    def train(self, epochs):

        loss_fn = CrossEntropyLoss()
        optim = Adagrad(params=(list(self.encoder.parameters()) + list(self.decoder.parameters())), lr=LEARNING_RATE)
        
        for epoch in range(1, epochs + 1):

            print(f"---------------EPOCH: {epoch}-------------")

            prev_state = self.decoder.init_hidden(BATCH_SIZE)

            print("Starting Training: ")
            train_loss = self.train_once(epoch, loss_fn, optim, prev_state)
            self.loss.append(train_loss)
            print("Epoch Loss: {}".format(train_loss))
            print("Epoch Training done")

            print("Starting Validation: ")
            val_loss_, cur_bleu = self.validate(loss_fn)
            self.val_loss.append(val_loss_)
            self.bleu_scores.append(cur_bleu)
            print("Epoch Validation done")

            if cur_bleu > self.blue_best:
                print(f"BLUE improved: {self.blue_best} to {cur_bleu}") 
                self.blue_best = cur_bleu
                self.save_model(epoch, optim)
            else:
                print("NO BLEU improved")
            
            print(f"---------------EPOCH: {epoch} Completed----------")
    
    def train_once(self, epoch, loss_fn, optim, prev_state):
        
        self.encoder.train()
        self.decoder.train()

        loss = 0
        (h_state, c_state) = prev_state
        print("No of training batches: {}".format(self.num_train_batches))
        for batch in range(1, self.num_train_batches + 1):
            
            # print("\tBatch: {}".format(batch))

            indices = self.train_loader.dataset.get_ids()
            new_sampler = SubsetRandomSampler(indices=indices)
            self.train_loader.batch_sampler.sampler = new_sampler

            images, captions = next(iter(self.train_loader))

            self.encoder.zero_grad()
            self.decoder.zero_grad()

            features = self.encoder(images)
            output, _ = self.decoder(captions, features)

            output = reshape(output, (-1, self.train_loader.dataset.vocab.get_vocab_size()))
            
            # h_state.detach_()
            # c_state.detach_()
            
            batch_loss = loss_fn(output, captions.view(-1))

            # optim.zero_grad()
            batch_loss.backward()
            optim.step()

            loss += batch_loss.item()

            # print("\tCurrent Batch loss: {}".format(batch_loss.item()))

            # print("Chumma val")
            # val_loss_, cur_bleu = self.validate(loss_fn)

            if batch % 50 == 0:
                print("Stats: Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch, batch_loss.item()))
            
        return loss / self.num_train_batches
    
    def validate(self, loss_fn):
        self.encoder.eval()
        self.decoder.eval()

        loss = 0.0
        bleu = 0.0

        # hidden = None
        with no_grad():
            for batch in range(1, self.num_val_batches + 1): # should be replaced with self.num_val_batches

<<<<<<< HEAD
                # print("\tBatch: {}".format(batch))
=======
            # print("\tBatch: {}".format(batch))
>>>>>>> temp

                indices = self.val_loader.dataset.get_ids()
                new_sampler = SubsetRandomSampler(indices=indices)
                self.val_loader.batch_sampler.sampler = new_sampler

                images, captions = next(iter(self.val_loader))

                # self.encoder.zero_grad()
                # self.decoder.zero_grad()

                features = self.encoder(images)
                output, _ = self.decoder(captions, features)
                
                cur_bleu = 0
                caption = captions[:,:-1]

                for index, _ in enumerate(output):
                    predicted_int = []
                    for scores in output[index]:
                        predicted_int.append(scores.argmax().item())
                    predicted_caption = self.train_loader.dataset.vocab.get_caption(predicted_int)
                    target_caption = self.train_loader.dataset.vocab.get_caption(caption[index].tolist())
                    cur_bleu +=  sentence_bleu([target_caption], predicted_caption, smoothing_function=SmoothingFunction().method1)

                bleu += cur_bleu / len(output)
                output = reshape(output, (-1, self.train_loader.dataset.vocab.get_vocab_size()))

                batch_loss = loss_fn(output, captions.view(-1))
                loss += batch_loss.item()

                # output.detach_()
                # hidden[0].detach_()
                # hidden[1].detach_()

        val_loss = loss / self.num_val_batches
        val_bleu = bleu / self.num_val_batches

        print("\tval_loss: {}; val_bleu: {}".format(val_loss, val_bleu))

        return val_loss, val_bleu 

    def save_model(self, epoch, optimizer):
        filename = os.path.join(MODEL_DIR, 'epoch-{}.pkl'.format(epoch))
        save({
            "cnn": self.encoder.state_dict(),
            "rnn": self.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": self.loss,
            "val_loss": self.val_loss,
            "val_bleu": self.bleu_scores,
            "epoch": epoch
        }, filename)
    
    def get_data_loader(self, root_dir, captions_file_name, instances_ann_name, image_dir, mode, transform):
        
        dataset = CustomDataset(root_dir, captions_file_name, instances_ann_name, image_dir, mode, transform)
        print("Dataset Created.")
        
        val_split = TRAIN_VAL_SPLIT
        indices = dataset.get_ids()

        split_indices = int(np.floor(val_split * len(indices)))

        np.random.seed(SEED)
        np.random.shuffle(indices)

        val_indices, train_indices = indices[split_indices:], indices[:split_indices]
        # val_indices, train_indices = indices[:100], indices[100:200]
    
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, sampler=train_sampler)

        val_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
        print("val train split done.")

        return (train_loader, val_loader)
    
    def load_model(self, path):
        checkpoint = load(path)
        self.encoder.load_state_dict(checkpoint['cnn'])
        self.decoder.load_state_dict(checkpoint['rnn'])

    def predict_using_sampling(self, image):
        image = Image.open(image)
        image = self.train_loader.dataset.transform(image).cuda()
        image = unsqueeze(image, 0)
        print(image.shape)
        features = self.encoder(image)
        features = unsqueeze(features, 0)
        print(features.shape)
        caption_idss = self.decoder.predict(features)

        for caption_ids in caption_idss:
            caption = self.train_loader.dataset.vocab.get_caption(caption_ids)
            print(caption)