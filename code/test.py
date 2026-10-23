import os
import torch
import utils
import lightning.pytorch as L
from model import G4SAINT
from datasets import ProteinDataModule
from lightning.pytorch.loggers import TensorBoardLogger

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
datamodeule.setup('train_val_test', act=True) # 'train_test'
test_loader = datamodeule.test_dataloader()

tb_logger = TensorBoardLogger("code/tb_logs", name='test', version=0) 

trainer = L.Trainer(logger=[tb_logger], 
                    max_epochs=70, accelerator=accelerator,
                    )

best_model_path = "code/model/g4saint/g4saint_predictor.ckpt" # g4saint
test_model = G4SAINT.load_from_checkpoint(best_model_path, strict=False)
# test_model.params["save_outputs"] = True
# test_model.params["save_fig"] = True
trainer.test(test_model, test_loader)
