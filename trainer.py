from tqdm import tqdm
import torch


class Trainer:

    def __init__(self, model, optimizer, criterion, device, scheduler, model_type):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.model_type = model_type

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        for batch in tqdm(loader, desc="Training"):

            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Check rõ ràng loại model thay vì check key trong dict
            if self.model_type in ["hybrid", "phobert_charcnn"]:
                char_in = batch["char_input"].to(self.device)
                logits = self.model(input_ids, mask, char_in)
            else:
                logits = self.model(input_ids, mask)

            loss = self.criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def eval_epoch(self, loader):

        self.model.eval()

        preds = []
        labels_all = []
        total_val_loss = 0

        with torch.no_grad():

            for batch in tqdm(loader, desc="Evaluating"):

                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                if self.model_type in ["hybrid", "phobert_charcnn"]:
                    char_in = batch["char_input"].to(self.device)
                    logits = self.model(input_ids, mask, char_in)
                else:
                    logits = self.model(input_ids, mask)

                # TÍNH VAL LOSS
                loss = self.criterion(logits, labels)
                total_val_loss += loss.item()

                pred = torch.argmax(logits, dim=1)

                preds.extend(pred.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
        avg_val_loss = total_val_loss / len(loader)
        return labels_all, preds, avg_val_loss