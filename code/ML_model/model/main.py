import os
import wandb

from data import load_dataset, batch_preparation_ctc
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from model_manager import get_model, LighntingE2EModelUnfolding
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def main(train_path="/root/model_training/grandstaff/partitions/train.txt",
         val_path="/root/model_training/grandstaff/partitions/val.txt",
         test_path="/root/model_training/grandstaff/partitions/val.txt",
         encoding="krn", 
         model_name="CRNN"):
    outpath = f"./out"
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(f"{outpath}/hyp", exist_ok=True)
    os.makedirs(f"{outpath}/gt", exist_ok=True)


    train_dataset, val_dataset, test_dataset = load_dataset(train_path, val_path, test_path, 
                                                            corpus_name=f"GrandStaff_{encoding}")

    _, i2w = train_dataset.get_dictionaries()

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=6, collate_fn=batch_preparation_ctc, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=6, collate_fn=batch_preparation_ctc)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=6, collate_fn=batch_preparation_ctc)

    maxheight, maxwidth = train_dataset.get_max_hw()

    lightning_model, torch_model = get_model(maxwidth=maxwidth, maxheight=maxheight, in_channels=1, 
                                  blank_idx=len(i2w), out_size=train_dataset.vocab_size()+1, 
                                  i2w=i2w, model_name=model_name, output_path=outpath)

    wandb_logger = WandbLogger(project='E2E_Pianoform', name=model_name)

    early_stopping = EarlyStopping(monitor='val_SER', min_delta=0.1, patience=5, mode="min", verbose=False)

    checkpointer = ModelCheckpoint(dirpath=f"./weights/", filename='CRNN-{epoch:02d}-{val_SER:.2f}',
                                   monitor="val_SER", mode='min', save_top_k=5, verbose=True)

    trainer = Trainer(max_epochs=25, logger=wandb_logger, callbacks=[checkpointer, early_stopping])

    trainer.fit(lightning_model, train_dataloader, val_dataloader)

    model = LighntingE2EModelUnfolding.load_from_checkpoint(checkpointer.best_model_path, model=torch_model)
    trainer.test(model, test_dataloader)
    wandb.finish()

if __name__ == "__main__":
    main()