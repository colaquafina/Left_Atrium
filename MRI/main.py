import os
import time
import glob
import sys

import torch
import numpy as np
import nibabel as nib
from torch import optim
from torch.backends import cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from loaddata import LoadDataset_scar, ProcessTestDataset
from network import Seg_3DNet_2task
from function import (
    F_LoadParam,
    F_loss_scar,
    F_mkdir,
    binary_dice_score,
    save_training_preview,
)

Root_DIR = '/work/users/g/s/gsonw/BIOS740/final_project/AJSQ_2/'
TRAIN_SAVE_DIR_best = Root_DIR + 'Script_AJSQnet/best_model/'
lossdir = Root_DIR + 'lossfile/'
os.makedirs(TRAIN_SAVE_DIR_best, exist_ok=True)
os.makedirs(lossdir, exist_ok=True)

lossfile1 = lossdir + 'laLoss_3d.txt'
lossfile2 = lossdir + 'scarLoss_3d.txt'
lossfile11 = lossdir + 'laLoss_3d_sdm.txt'
lossfile21 = lossdir + 'scarMaskLoss_1.txt'
lossfile22 = lossdir + 'scarMaskLoss_2.txt'
lossfile_la_dice = lossdir + 'laDice_3d.txt'
lossfile_scar_dice = lossdir + 'scarDice_3d.txt'
preview_dir = Root_DIR + 'training_previews/'
os.makedirs(preview_dir, exist_ok=True)

WORKERSNUM = 16
BatchSize = 2
NumEPOCH = 600
LEARNING_RATE = 1e-3
REGULAR_RATE = 0.96
PREVIEW_SAVE_INTERVAL = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingDataset(data.Dataset):
    def __init__(self, subject_files, augment=False):
        """
        subject_files: list of subject folders
        augment: whether to apply training augmentation
        """
        self.datafile = subject_files
        self.augment = augment

    def __getitem__(self, item):
        subject_path = self.datafile[item]

        imagename = os.path.join(subject_path, 'enhanced.nii.gz')
        LAlabelname = os.path.join(subject_path, 'atriumSegImgMO.nii.gz')
        LAscarMaplabelname = os.path.join(subject_path, 'scarSegImgM.nii.gz')

        numpyimage, numpylabel_LA, numpylabel_LAdist, numpyprob_normal, numpyprob_scar = LoadDataset_scar(
            imagename,
            LAlabelname,
            LAscarMaplabelname,
            augment=self.augment
        )

        numpyimage = np.array(numpyimage)
        numpylabel_LA = np.array(numpylabel_LA)
        numpylabel_LA = (numpylabel_LA > 0) * 1
        numpylabel_LAdist = np.array(numpylabel_LAdist)
        numpyprob_normal = np.array(numpyprob_normal)
        numpyprob_scar = np.array(numpyprob_scar)

        tensorimage = torch.from_numpy(numpyimage).float()
        tensorlabel_LA = torch.from_numpy(numpylabel_LA.astype(np.float32))
        tensorlabel_LAdist = torch.from_numpy(numpylabel_LAdist.astype(np.float32))
        tensorprob_normal = torch.from_numpy(numpyprob_normal.astype(np.float32))
        tensorprob_scar = torch.from_numpy(numpyprob_scar.astype(np.float32))

        return tensorimage, tensorlabel_LA, tensorlabel_LAdist, tensorprob_normal, tensorprob_scar

    def __len__(self):
        return len(self.datafile)


def Train_process(dataload, net, epoch, optimizer):
    start_time = time.time()
    flearning_rate = LEARNING_RATE * (REGULAR_RATE ** (epoch // 10))
    if flearning_rate < 1e-5:
        flearning_rate = 1e-5

    net.train()

    total_loss = 0.0
    total_la_loss = 0.0
    total_sdf_loss = 0.0
    total_scar_loss = 0.0
    total_scar_m1_loss = 0.0
    total_scar_m2_loss = 0.0
    total_la_dice = 0.0
    total_scar_dice = 0.0
    total_samples = 0
    strNetSaveName = 'not_saved_this_epoch'

    for i, (lgeimage, lgelabel, lgedist, lgeprob_normal, lgeprob_scar) in enumerate(dataload):
        for param_group in optimizer.param_groups:
            param_group['lr'] = flearning_rate

        lgeimage = lgeimage.to(device)
        lgelabel = lgelabel.to(device)
        lgedist = lgedist.to(device)
        lgeprob_normal = lgeprob_normal.to(device)
        lgeprob_scar = lgeprob_scar.to(device)

        optimizer.zero_grad()

        output = net(lgeimage)

        loss_la, loss_sdf_la, loss_scar,loss_scar_m1, loss_scar_m2 = F_loss_scar(
            output, lgelabel, lgedist, lgeprob_normal, lgeprob_scar
        )

        loss = loss_la + 0.01 * loss_sdf_la + loss_scar + 0.01 * loss_scar_m1 + 0.001*loss_scar_m2

        loss.backward()
        optimizer.step()

        out_la, out_scar = output
        batch_la_dice = binary_dice_score(out_la, lgelabel)
        batch_scar_dice = binary_dice_score(out_scar[:, 1:2], lgeprob_scar)
        total_la_dice += batch_la_dice.sum().item()
        total_scar_dice += batch_scar_dice.sum().item()
        total_samples += batch_la_dice.numel()

        if i == 0 and epoch % PREVIEW_SAVE_INTERVAL == 0:
            save_training_preview(
                epoch,
                lgeimage,
                lgelabel,
                lgeprob_scar,
                out_la,
                out_scar,
                tag='train',
                preview_dir=preview_dir
            )

        total_loss += loss.item()
        total_la_loss += loss_la.item()
        total_sdf_loss += loss_sdf_la.item()
        total_scar_loss += loss_scar.item()
        total_scar_m1_loss += loss_scar_m1.item()
        total_scar_m2_loss += loss_scar_m2.item()

    num_batches = len(dataload)

    avg_loss = total_loss / num_batches
    avg_la_loss = total_la_loss / num_batches
    avg_sdf_loss = total_sdf_loss / num_batches
    avg_scar_loss = total_scar_loss / num_batches
    avg_scar_m1_loss = total_scar_m1_loss / num_batches
    avg_scar_m2_loss = total_scar_m2_loss / num_batches
    avg_la_dice = total_la_dice / total_samples
    avg_scar_dice = total_scar_dice / total_samples

    with open(lossfile1, 'a') as f1:
        f1.write(f"{epoch},{avg_la_loss}\n")

    with open(lossfile2, 'a') as f2:
        f2.write(f"{epoch},{avg_scar_loss}\n")

    with open(lossfile11, 'a') as f11:
        f11.write(f"{epoch},{avg_sdf_loss}\n")

    with open(lossfile21, 'a') as f21:
        f21.write(f"{epoch},{avg_scar_m1_loss}\n")

    with open(lossfile22, 'a') as f22:
        f22.write(f"{epoch},{avg_scar_m2_loss}\n")

    with open(lossfile_la_dice, 'a') as f_la_dice:
        f_la_dice.write(f"{epoch},{avg_la_dice}\n")

    with open(lossfile_scar_dice, 'a') as f_scar_dice:
        f_scar_dice.write(f"{epoch},{avg_scar_dice}\n")

    print(
        'Train epoch %d, avg loss: %.6f, avg LA Dice: %.4f, avg scar Dice: %.4f'
        % (epoch, avg_loss, avg_la_dice, avg_scar_dice)
    )

    end_time = time.time()
    print(
        '---------------- Train Seg-Net: %s, epoch %d cost time: %3.2f ----------------'
        % (strNetSaveName, epoch, end_time - start_time)
    )

    return avg_loss

def Validate(dataload, net, epoch):
    net.eval()

    total_loss = 0.0
    total_la_loss = 0.0
    total_scar_loss = 0.0
    total_la_dice = 0.0
    total_scar_dice = 0.0
    total_samples = 0
    scar_true_positives = 0
    scar_false_positives = 0
    scar_false_negatives = 0

    with torch.no_grad():
        for i, (lgeimage, lgelabel, lgedist, lgeprob_normal, lgeprob_scar) in enumerate(dataload):

            lgeimage = lgeimage.to(device)
            lgelabel = lgelabel.to(device)
            lgedist = lgedist.to(device)
            lgeprob_normal = lgeprob_normal.to(device)
            lgeprob_scar = lgeprob_scar.to(device)

            output = net(lgeimage)

            loss_la, loss_sdf_la, loss_scar, loss_scar_m1, loss_scar_m2 = F_loss_scar(
                output,
                lgelabel,
                lgedist,
                lgeprob_normal,
                lgeprob_scar
            )

            loss = loss_la + 0.01 * loss_sdf_la + loss_scar + 0.01 * loss_scar_m1 + 0.001*loss_scar_m2

            out_la, out_scar = output
            batch_la_dice = binary_dice_score(out_la, lgelabel)
            batch_scar_dice = binary_dice_score(out_scar[:, 1:2], lgeprob_scar)
            total_la_dice += batch_la_dice.sum().item()
            total_scar_dice += batch_scar_dice.sum().item()
            total_samples += batch_la_dice.numel()

            predicted_scar = out_scar[:, 1:2] > 0.5
            target_scar = lgeprob_scar > 0.5
            scar_true_positives += torch.sum(predicted_scar & target_scar).item()
            scar_false_positives += torch.sum(predicted_scar & ~target_scar).item()
            scar_false_negatives += torch.sum(~predicted_scar & target_scar).item()

            if i == 0 and epoch % PREVIEW_SAVE_INTERVAL == 0:
                save_training_preview(
                    epoch,
                    lgeimage,
                    lgelabel,
                    lgeprob_scar,
                    out_la,
                    out_scar,
                    tag='validation',
                    preview_dir=preview_dir
                )

            total_loss += loss.item()
            total_la_loss += loss_la.item()
            total_scar_loss += loss_scar.item()

    avg_loss = total_loss / len(dataload)
    avg_la_loss = total_la_loss / len(dataload)
    avg_scar_loss = total_scar_loss / len(dataload)
    avg_la_dice = total_la_dice / total_samples
    avg_scar_dice = total_scar_dice / total_samples
    scar_precision = scar_true_positives / max(
        scar_true_positives + scar_false_positives,
        1,
    )
    scar_recall = scar_true_positives / max(
        scar_true_positives + scar_false_negatives,
        1,
    )

    print(
        'Validation epoch %d, loss: %.6f, avg LA Dice: %.4f, '
        'avg scar Dice: %.4f, scar precision: %.4f, scar recall: %.4f, '
        'avg scar loss: %.4f'
        % (
            epoch,
            avg_loss,
            avg_la_dice,
            avg_scar_dice,
            scar_precision,
            scar_recall,
            avg_scar_loss,
        )
    )

    return avg_loss


TRAIN_DIR_PATH = '/work/users/g/s/gsonw/BIOS740/final_project/LA_scar_quantification/train_data'
TEST_DIR_PATH = '/work/users/g/s/gsonw/BIOS740/final_project/LA_scar_segmentation/val_data'


def initialize_log_files():
    headers = {
        lossfile1: "epoch,avg_la_loss\n",
        lossfile2: "epoch,avg_scar_loss\n",
        lossfile11: "epoch,avg_sdf_loss\n",
        lossfile21: "epoch,avg_scar_mask_loss_1\n",
        lossfile22: "epoch,avg_scar_mask_loss_2\n",
        lossfile_la_dice: "epoch,avg_la_dice\n",
        lossfile_scar_dice: "epoch,avg_scar_dice\n",
    }
    for path, header in headers.items():
        with open(path, 'w') as log_file:
            log_file.write(header)


def create_data_loaders():
    all_subjects = sorted(glob.glob(os.path.join(TRAIN_DIR_PATH, '*')))
    train_subjects, val_subjects = train_test_split(
        all_subjects,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    print('Number of training subjects:', len(train_subjects))
    print('Number of validation subjects:', len(val_subjects))

    loader_options = {
        'batch_size': BatchSize,
        'num_workers': WORKERSNUM,
        'pin_memory': torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        TrainingDataset(train_subjects, augment=True),
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        TrainingDataset(val_subjects, augment=False),
        shuffle=True,
        **loader_options,
    )
    return train_loader, val_loader


def train_model():
    print('training')
    net = Seg_3DNet_2task(1, 1).to(device)
    train_loader, val_loader = create_data_loaders()
    cudnn.benchmark = torch.cuda.is_available()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE,  weight_decay=1e-4)
    initialize_log_files()

    best_val_loss = float('inf')
    best_epoch = -1
    patience = 30
    min_delta = 1e-4
    epochs_no_improve = 0

    for epoch in range(NumEPOCH):
        Train_process(train_loader, net, epoch, optimizer)
        val_loss = Validate(val_loader, net, epoch)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_path = os.path.join(TRAIN_SAVE_DIR_best, 'best_net.pkl')
            torch.save(net.state_dict(), best_model_path)
            print('Saved best model at epoch %d with val loss %.6f' % (epoch, val_loss))
            continue

        epochs_no_improve += 1
        # print(
        #     'No improvement for %d/%d epochs. Best epoch: %d, best val loss: %.6f'
        #     % (epochs_no_improve, patience, best_epoch, best_val_loss)
        # )
        # # if epochs_no_improve >= patience:
        #     print(
        #         'Early stopping at epoch %d. Best epoch: %d, best val loss: %.6f'
        #         % (epoch, best_epoch, best_val_loss)
        #     )
        #     break


def run_inference():
    print('testing ....')
    net = Seg_3DNet_2task(1, 1).to(device)
    checkpoint_path = os.path.join(TRAIN_SAVE_DIR_best, 'best_net.pkl')
    print('Loading checkpoint:', checkpoint_path)
    F_LoadParam(checkpoint_path, net)
    net.eval()

    for subject_path in sorted(glob.glob(os.path.join(TEST_DIR_PATH, '*'))):
        image_path = os.path.join(subject_path, 'enhanced.nii.gz')
        predict_la, predict_scar = ProcessTestDataset(image_path, net)
        save_dir = subject_path.replace('val_data', 'val_data_result')
        F_mkdir(save_dir)
        nib.save(predict_la, os.path.join(save_dir, 'LA_predict_AJSQnet_SESA.nii.gz'))
        nib.save(predict_scar, os.path.join(save_dir, 'scar_predict_AJSQnet_SESA.nii.gz'))

    print('testing end')


def main():
    action = sys.argv[1].lower() if len(sys.argv) > 1 else 'train'
    if action == 'train':
        train_model()
    elif action in {'test', 'inference'}:
        run_inference()
    else:
        raise ValueError("Action must be 'train', 'test', or 'inference'.")


if __name__ == '__main__':
    main()
