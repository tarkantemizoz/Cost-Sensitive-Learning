import time
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

class Optimization:
    """ A helper class to train, test and diagnose Cost-sensitive Logistic Regression
        
    Attributes:
        model: CSLR model.
        optimizer: Optimizer of the network.
        train_return: List of train returns.
        val_return: List of validation returns.
        validation: Whether there is validation data.
        batch_size: Batch-size of the network.
        n_epochs: Total number of epochs.
        n_steps: Number of epochs to evaluate the results
    """
    
    def __init__(self, model, optimizer, config):
        """Initialises CLSR.
            
        Args:
            model: CSLR model.
            optimizer: Optimizer of the network.
            config: Configuration of the network.
        """
        
        self.model = model
        self.optimizer = optimizer
        self.train_return = []
        self.val_return = []
        self.weight_norm = []
        self.grad_norm = []
        self.validation = False
        self.batch_size = config.get("batch_size",32)
        self.n_epochs = config.get("n_epochs", 1000)
        self.n_steps = config.get("n_steps", self.n_epochs)
    
    @staticmethod
    def batch(iterable, n):
        """Creates batches."""
        
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]        
            
    def train(self, x_train, y_train, r_train, x_val=None, y_val=None, r_val=None):
        """Applies simple feed-forward network to an input.
        
        Args:
            x_train: train features
            r_train: train returns
            x_val: validation features
            r_val: validation returns
        """
        
        if x_val is not None or r_val is not None:
            self.validation = True
        start_time = time.time()

        for epoch in range(self.n_epochs):
            x_shuff, r_shuff, y_shuff = shuffle(x_train, r_train, y_train)
            self.model.train()
            
            #TODO: Uncomment this to see the effect of the scaling
            if torch.min(r_shuff) < 0:
                r_shuff = r_shuff - torch.min(r_shuff)
            r_shuff = r_shuff / torch.max(r_shuff)

            for j in self.batch(range(0, len(x_shuff)),self.batch_size):
            #for i, (x_batch, y_batch, r_batch) in enumerate(train_loader):
                if len(j) < 32:
                    break
                x_batch = x_shuff[j]
                r_batch = r_shuff[j]
                y_batch = torch.tensor(y_shuff[j], dtype=torch.long)

                self.optimizer.zero_grad()
                outputs, probs, logits = self.model(x_batch)

                #TODO: try learning with cross entropy loss
                #ce_loss = torch.nn.CrossEntropyLoss()
                #loss = ce_loss(probs, y_batch)
                #loss.backward()

                loss = -(torch.mul(outputs, r_batch)).sum()
                loss.backward()

                print_weights = False
                print_grads = False
                print_clips = False
                clipping = False

                self.weight_norm.append(torch.sum(torch.absolute(self.model.layer_0.weight)).detach().numpy().item())
                grad_norm = torch.sqrt(torch.sum(self.model.layer_0.weight.grad * self.model.layer_0.weight.grad))
                self.grad_norm.append(np.log(grad_norm.detach().numpy().item()))

                if print_weights == True and ((epoch+1) % self.n_steps == 0):
                    print("weights in train")
                    print(self.model.layer_0.weight)
                    print("weights norm")
                    print(torch.sum(torch.absolute(self.model.layer_0.weight)))

                if print_grads == True and ((epoch+1) % self.n_steps == 0):
                    print("gradients in train")
                    print(self.model.layer_0.weight.grad)
                    print(outputs)

                    #gradients = torch.zeros((r_batch.shape[1], x_batch.shape[1]))
                    #for i in range(r_batch.shape[1]):                    
                    #    first_part = (torch.sum(r_batch * outputs, dim=1) - r_batch[:,i]) * outputs[:,i] / self.model.temp
                    #    for j in range(x_batch.shape[1]): 
                    #        gradients[i,j] = torch.sum(first_part * x_batch[:,j])
                    #print("gradients self calculation")
                    #print(gradients)

                    #test_clipping = np.zeros((r_batch.shape[1], x_batch.shape[1]))
                    #grad_norm = torch.sqrt(torch.sum(gradients * gradients))
                    #clipped_gradients = 0.1 * gradients / grad_norm
                    #print("gradients clipping self calculation")
                    #print(clipped_gradients)

                if clipping == True:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1, 2)             
                    if print_clips == True and ((epoch+1) % self.n_steps == 0):
                        print("gradients in after clipping")
                        print(self.model.layer_0.weight.grad)
                        
                self.optimizer.step()
            
            returns_train, _, _ = self.evaluate(x_train, r_train)
            self.train_return.append(returns_train)
            if self.validation is True:
                returns_val, _, _ = self.evaluate(x_val, r_val)
                self.val_return.append(returns_val)

            if ((epoch+1) % self.n_steps == 0):
                elapsed = time.time() - start_time               
                print(
                    ("Epoch %d Train Return: %.3f.")  % (epoch + 1, self.train_return[-1]),
                    ((" Validation Return: %.3f. Elapsed time: %.3fs.")
                     % (self.val_return[-1], elapsed)
                     if self.validation is True else 
                     " Elapsed time: %.3fs."
                     % elapsed) 
                )
                start_time = time.time()          

    def evaluate(self, x_test, r_test):
        """Evaluates simple feed-forward network to an input.
            
        Args:
            x_test: features of the evaluated data
            r_test: returns of the evaluated data
            
        Returns:
            Triple of Tensors for: (Total returns, decision variables, probabilities)
        """
        
        with torch.no_grad():
            outputs, probs, _ = self.model(x_test)
            returns = torch.mul(outputs, r_test).sum()
            
            return returns, outputs, probs           
           



                
                
                             
                            

