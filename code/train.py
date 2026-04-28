import os
import torch
import utils
import lightning.pytorch as L
from model import G4SAINT
from datasets import ProteinDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

os.chdir(os.path.expanduser("~"))
device = utils.get_available_gpu()
accelerator = 'cpu' if str(device) == 'cpu' else 'gpu'
torch.set_float32_matmul_precision('high')

datamodeule = ProteinDataModule(task_type='pretrain', 
                                y_type = 'str',
                                concat = True,
                                balance = 'no',  
                                batch_size=64, num_workers=0,
                                model_name = 'esm2_t33_650M_UR50D',
                                layer_num = 33,
                                hub_dir = 'code/esm2_t33_650M_UR50D',
                                )
datamodeule.setup('train_val_test', act=True)# 'train_val_test''train_test'
train_loader = datamodeule.train_dataloader()
val_loader = datamodeule.val_dataloader()
test_loader = datamodeule.test_dataloader()

# tb_logger = TensorBoardLogger("code/tb_logs", name='train', version=0) # pre

use_domain = True
use_sae = False
params = utils.get_default_params(alpha_dict=True)
params.update({
    "hidden_channels": 512,
    "num_layers": 2,
    "attn_order": "domain_first",
    "optimizer": "AdamW",
    "lr": 2.74461E-05, 
    "weight_decay": 3E-06, 
    "scheduler": "CosineAnnealingLR",
    "use_domain": use_domain,
    "use_sae": use_sae,
})
prefix = ""
if params.get("use_sae"):
    prefix += "s_"
if params.get("use_domain"):
    prefix += "d_"

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='code/model/train',
    filename=prefix+'{epoch:02d}_{val_loss:.3f}',
    save_top_k=1,
    mode='min'
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

model = G4SAINT(params)
trainer = L.Trainer(logger=False, # [tb_logger],
                    max_epochs=70, accelerator=accelerator,
                    callbacks=[checkpoint_callback, early_stopping_callback], 
                    )

trainer.fit(model, train_loader, val_loader)

best_model_path = checkpoint_callback.best_model_path
print(f"Best model path: {best_model_path}")
test_model = G4SAINT.load_from_checkpoint(best_model_path, strict=False)
trainer.test(test_model, test_loader)
