
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from utils import calculate_acc


class Train :
    def __init__(self, 
                model,
                num_epoch,
                optimizer,
                scheduler,
                criterion,
                tr_loader,
                val_loader,
                ) :

        self.model = model
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.tr_loader = tr_loader
        self.val_loader = val_loader
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def training(self) :
        print('🚀 Start Trainiing... 🚀')
        print(f'Using Resource is... : {self.device}')
        best_acc = 0.0

        early_count = 0
        best_val_loss = 1000

        for epoch in range(1, self.num_epoch + 1) :
            start = time.time()
            
            train_loss, train_acc = self.train()
            val_loss, val_acc = self.eval(phase='valid')
                        
            it_takes = time.time() - start


            if best_val_loss > val_loss:
                best_val_loss = val_loss

            elif best_val_loss < val_loss:
                early_count += 1
                

            print(f'EPOCH {epoch}/{self.num_epoch}\tEARLY STACK {early_count}/5')
            print(f'TRAIN LOSS : {train_loss:.3f}\tTRAIN ACC : {train_acc*100:.2f}%')
            print(f'VALIDATION LOSS : {val_loss:.3f}\tVALIDATION ACC : {val_acc*100:.2f}%')
            print(f'It takes... {it_takes/60:.2f}m')
            

            if val_acc > best_acc :
                best_acc = val_acc 
                print(f'\n✅ BEST MODEL IS SAVED at {epoch} epoch')
                torch.save(self.model.state_dict(), './BEST_MODEL.pt')

            print("LR Scheduler is Working..")
            self.scheduler.step(val_loss)

            if early_count == 5:
                print("Early Stopping!")
                break


            print('=='*40)

        # print('\n\n')
        # print('✨ Start Evaluation... ✨')
        # test_loss, test_acc = self.eval(phase='test')
        # print(f'TEST LOSS : {test_loss:.3f}\tTEST ACC : {test_acc*100:.2f}%')


    def train(self) :
        epoch_loss, epoch_acc = 0, 0
        self.model.train()
        self.model.train_mode = True
        self.model.to(self.device)

        for imgs, tabular , labels in tqdm(self.tr_loader) :
            imgs, tabular, labels = imgs.to(self.device), tabular.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            y_preds = self.model(imgs, tabular)

            loss = self.criterion(y_preds, labels)
            acc = calculate_acc(y_preds, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss/len(self.tr_loader), epoch_acc/len(self.tr_loader)


    def eval(self, phase="valid") : 
        epoch_loss, epoch_acc = 0, 0
        self.model.eval()

        with torch.no_grad() :

            self.model.to(self.device)

            for imgs, tabular, labels in self.val_loader :
                imgs, tabular, labels = imgs.to(self.device), tabular.to(self.device), labels.to(self.device)
                y_preds = self.model(imgs, tabular)
                
                loss = self.criterion(y_preds, labels)
                acc = calculate_acc(y_preds, labels)

                epoch_loss += loss.item()
                epoch_acc += acc.item()


        return epoch_loss/len(self.val_loader), epoch_acc/len(self.val_loader)

