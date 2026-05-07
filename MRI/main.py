import os
import time
import glob
import torch
import numpy as np
import nibabel as nib
import sys
from torch import optim
from torch.nn import DataParallel
from torch.backends import cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
# import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from loaddata import LoadDataset_scar, ProcessTestDataset, weak_intensity_aug, strong_intensity_aug
from network import Seg_3DNet, Seg_3DNet_2task
from function import F_loss_scar, F_LoadParam, F_mkdir, consistency_loss, get_consistency_weight, binary_dice_score, save_training_preview, _find_latest_checkpoint,binary_dice_score_scar

#Root_DIR = '/home/lilei/MICCAI2020/Data60/'
Root_DIR = '/work/users/g/s/gsonw/BIOS740/final_project/AJSQ_2/'
TRAIN_SAVE_DIR_best = Root_DIR + 'Script_AJSQnet/best_model/'
TRAIN_SAVE_DIR_best_CON = Root_DIR + 'Script_AJSQnet_consistency/best_model/'
lossdir = Root_DIR + 'lossfile/'
os.makedirs(TRAIN_SAVE_DIR_best, exist_ok=True)
os.makedirs(lossdir, exist_ok=True)
os.makedirs(TRAIN_SAVE_DIR_best_CON, exist_ok=True)

lossfile1 = lossdir + 'laLoss_3d.txt'
lossfile2 = lossdir + 'scarLoss_3d.txt'
lossfile11 = lossdir + 'laLoss_3d_sdm.txt'
lossfile21 = lossdir + 'scarMaskLoss_1.txt'
lossfile22 = lossdir + 'scarMaskLoss_2.txt'
lossfile_cons = lossdir + 'consistencyLoss_3d.txt'
lossfile_la_dice = lossdir + 'laDice_3d.txt'
lossfile_scar_dice = lossdir + 'scarDice_3d.txt'
preview_dir = Root_DIR + 'training_previews/'
os.makedirs(preview_dir, exist_ok=True)

WORKERSNUM = 16
BatchSize = 2
NumEPOCH = 1200
LEARNING_RATE = 1e-3
REGULAR_RATE = 0.96
WEIGHT_DECAY = 1e-4
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

@torch.no_grad()
def update_ema_teacher(student, teacher, alpha=0.99):
    """
    teacher = alpha * teacher + (1 - alpha) * student
    """
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()

    for key in teacher_state.keys():
        teacher_state[key].copy_(
            teacher_state[key] * alpha + student_state[key] * (1.0 - alpha)
        )


def Train_process(dataload, net, epoch, optimizer, savedir):
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

        loss_la, loss_sdf_la, loss_scar, loss_scar_m1, loss_scar_m2 = F_loss_scar(
            output, lgelabel, lgedist, lgeprob_normal, lgeprob_scar
        )

        weight_sdm = 1e-2 * (1.05 ** (epoch // 10))
        weight_scar = 1e-2 * (1.05 ** (epoch // 10))

        loss = (loss_la + weight_sdm * loss_sdf_la + 10 * loss_scar + 0.01 * loss_scar_m1 + 0.01 * loss_scar_m2)

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

        # if i % 50 == 0:
        #     print(
        #         'epoch %d, batch %d, lr: %.10f, batch loss: %.10f'
        #         % (epoch, i, flearning_rate, loss.item())
        #     )

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

    # if epoch > NumEPOCH*0.95:
    #     strNetSaveName = 'net_with_%d.pkl' % epoch
    #     torch.save(net.state_dict(), os.path.join(savedir, strNetSaveName))

    end_time = time.time()
    print(
        '---------------- Train Seg-Net: %s, epoch %d cost time: %3.2f ----------------'
        % (strNetSaveName, epoch, end_time - start_time)
    )

    return avg_loss

def Train_process_consistency(dataload, student, teacher, epoch, optimizer, savedir):
    start_time = time.time()

    flearning_rate = LEARNING_RATE * (REGULAR_RATE ** (epoch // 10))
    if flearning_rate < 1e-5:
        flearning_rate = 1e-5

    student.train()
    teacher.eval()

    total_loss = 0.0
    total_la_loss = 0.0
    total_sdf_loss = 0.0
    total_scar_loss = 0.0
    total_scar_m1_loss = 0.0
    total_scar_m2_loss = 0.0
    total_cons_loss = 0.0
    total_la_dice = 0.0
    total_scar_dice = 0.0
    total_samples = 0

    cons_weight = get_consistency_weight(epoch)
    for i, (lgeimage, lgelabel, lgedist, lgeprob_normal, lgeprob_scar) in enumerate(dataload):
        for param_group in optimizer.param_groups:
            param_group['lr'] = flearning_rate
        lgeimage = lgeimage.to(device)
        lgelabel = lgelabel.to(device)
        lgedist = lgedist.to(device)
        lgeprob_normal = lgeprob_normal.to(device)
        lgeprob_scar = lgeprob_scar.to(device)

        weak_image = weak_intensity_aug(lgeimage.clone())
        strong_image = strong_intensity_aug(lgeimage.clone())
        optimizer.zero_grad()

        student_output = student(strong_image)

        with torch.no_grad():
            teacher_output = teacher(weak_image)

        loss_la, loss_sdf_la, loss_scar, loss_scar_m1, loss_scar_m2 = F_loss_scar(
            student_output,
            lgelabel,
            lgedist,
            lgeprob_normal,
            lgeprob_scar,
        )      
        weight_sdm = 1e-2 * (1.05 ** (epoch // 10))
        supervised_loss = (loss_la + weight_sdm * loss_sdf_la + 10 * loss_scar + 0.01 * loss_scar_m1 + 0.01 * loss_scar_m2)
        loss_cons, loss_cons_la, loss_cons_scar = consistency_loss(student_output, teacher_output, SCAR_CONS_WEIGHT=0.2)
        loss = supervised_loss + cons_weight * loss_cons
        loss.backward()
        optimizer.step()  
        update_ema_teacher(student, teacher, alpha=0.99)

        student_la, student_scar = student_output
        batch_la_dice = binary_dice_score(student_la, lgelabel)
        batch_scar_dice = binary_dice_score(student_scar[:, 1:2], lgeprob_scar, threshold=0.5)
        total_la_dice += batch_la_dice.sum().item()
        total_scar_dice += batch_scar_dice.sum().item()
        total_samples += batch_la_dice.numel()

        if i == 0 and epoch % PREVIEW_SAVE_INTERVAL == 0:
            save_training_preview(
                epoch,
                lgeimage,
                lgelabel,
                lgeprob_scar,
                student_la,
                student_scar,
                tag='consistency_train',
                preview_dir=preview_dir
            )

        total_loss += loss.item()
        total_la_loss += loss_la.item()
        total_sdf_loss += loss_sdf_la.item()
        total_scar_loss += loss_scar.item()
        total_scar_m1_loss += loss_scar_m1.item()
        total_scar_m2_loss += loss_scar_m2.item()
        total_cons_loss += loss_cons.item()
        
    num_batches = len(dataload)

    avg_loss = total_loss / num_batches
    avg_la_loss = total_la_loss / num_batches
    avg_sdf_loss = total_sdf_loss / num_batches
    avg_scar_loss = total_scar_loss / num_batches
    avg_scar_m1_loss = total_scar_m1_loss / num_batches
    avg_scar_m2_loss = total_scar_m2_loss / num_batches
    avg_cons_loss = total_cons_loss / num_batches
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

    with open(lossfile_cons, 'a') as fc:
        fc.write(f"{epoch},{avg_cons_loss},{cons_weight}\n")

    with open(lossfile_la_dice, 'a') as f_la_dice:
        f_la_dice.write(f"{epoch},{avg_la_dice}\n")

    with open(lossfile_scar_dice, 'a') as f_scar_dice:
        f_scar_dice.write(f"{epoch},{avg_scar_dice}\n")
    print(
        'Train epoch %d, avg loss: %.6f, avg LA Dice: %.4f, avg scar Dice: %.4f'
        % (epoch, avg_loss, avg_la_dice, avg_scar_dice)
    )

    if epoch > NumEPOCH * 0.95:
        strNetSaveName = 'student_net_with_%d.pkl' % epoch
        torch.save(student.state_dict(), os.path.join(savedir, strNetSaveName))

        strTeacherSaveName = 'teacher_net_with_%d.pkl' % epoch
        torch.save(teacher.state_dict(), os.path.join(savedir, strTeacherSaveName))

    end_time = time.time()
    print(
        '---------------- Train Mean-Teacher Seg-Net: epoch %d cost time: %3.2f ----------------'
        % (epoch, end_time - start_time)
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

            weight_sdm = 1e-2 * (1.05 ** (epoch // 10))
            loss = loss_la + weight_sdm * loss_sdf_la + 10 * loss_scar + 0.01 * loss_scar_m1 + 0.01 * loss_scar_m2

            out_la, out_scar = output
            batch_la_dice = binary_dice_score(out_la, lgelabel)
            batch_scar_dice = binary_dice_score_scar(out_scar[:, 1:2], lgeprob_scar, threshold=0.5)
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

    print(
        'Validation epoch %d, loss: %.6f, LA loss: %.6f, scar loss: %.6f, avg LA Dice: %.4f, avg scar Dice: %.4f'
        % (epoch, avg_loss, avg_la_loss, avg_scar_loss, avg_la_dice, avg_scar_dice)
    )

    return avg_loss

def main():
    is_for_training = True
    is_consistency = False
    TRAIN_DIR_PATH =  '/work/users/g/s/gsonw/BIOS740/final_project/LA_scar_quantification/train_data'
    TEST_DIR_PATH = '/work/users/g/s/gsonw/BIOS740/final_project/LA_scar_quantification/test_data'
    TRAIN_SAVE_DIR_Seg = Root_DIR + 'Script_AJSQnet/result_model/'
    TRAIN_SAVE_DIR_Seg_Con =  Root_DIR + 'Script_AJSQnet_consistency/result_model/'
    os.makedirs(TRAIN_SAVE_DIR_Seg, exist_ok=True)
    os.makedirs(TRAIN_SAVE_DIR_Seg_Con, exist_ok=True)

    if len(sys.argv) > 1:
        if sys.argv[1].find('train') != -1:
            is_for_training = True
        else:
            is_for_training = False

    if len(sys.argv) >2:
        if sys.argv[2].find('consistency') != -1:
            is_consistency = True
        else:
            is_consistency = False


    if is_for_training and not is_consistency:
        print('training')
        net: Seg_3DNet_2task[int, int] = Seg_3DNet_2task(1, 1).to(device)
        all_subjects = sorted(glob.glob(TRAIN_DIR_PATH + '/*'))

        train_subjects, val_subjects = train_test_split(all_subjects, test_size=0.2, random_state=42,shuffle=True)

        print("Number of training subjects:", len(train_subjects))
        print("Number of validation subjects:", len(val_subjects))

        train_dataset = TrainingDataset(train_subjects, augment=True)
        val_dataset = TrainingDataset(val_subjects, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True)

        val_loader = DataLoader(val_dataset, batch_size=BatchSize, shuffle=False,num_workers=WORKERSNUM,pin_memory=True)  
        
        cudnn.benchmark = True
        #net = DataParallel(net,device_ids=[0,2,3])

        optimizer = optim.Adam(net.parameters())
        #optimizer = optim.SGD(net.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

        # Seg_net_param = TRAIN_SAVE_DIR_Seg + 'net_with_99.pkl'
        # # Seg_net_param = TRAIN_SAVE_DIR_best + 'net_with_99.pkl'
        # F_LoadParam(Seg_net_param, net)

        # best_val_loss = float("inf")
        # best_epoch = -1

        # patience = 15          # stop if no improvement for 15 epochs
        # min_delta = 1e-4       # minimum improvement required
        # epochs_no_improve = 0
        
        headers = {
            lossfile1: "epoch,avg_la_loss\n",
            lossfile2: "epoch,avg_scar_loss\n",
            lossfile11: "epoch,avg_sdf_loss\n",
            lossfile21: "epoch,avg_scar_mask_loss_1\n",
            lossfile22: "epoch,avg_scar_mask_loss_2\n",
            lossfile_cons: "epoch,avg_consistency_loss,consistency_weight\n",
            lossfile_la_dice: "epoch,avg_la_dice\n",
            lossfile_scar_dice: "epoch,avg_scar_dice\n",
        }

        for f, header in headers.items():
            with open(f, 'w') as file:
                file.write(header)
        
        for epoch in range(NumEPOCH):
            train_loss = Train_process(train_loader, net, epoch, optimizer, TRAIN_SAVE_DIR_Seg)

            val_loss = Validate(val_loader, net, epoch)

            # if val_loss < best_val_loss - min_delta:
            #     best_val_loss = val_loss
            #     best_epoch = epoch
            #     epochs_no_improve = 0

            #     best_model_path = os.path.join(TRAIN_SAVE_DIR_best, 'best_net.pkl')
            #     torch.save(net.state_dict(), best_model_path)

            #     print("Saved best model at epoch %d with val loss %.6f" % (epoch, val_loss))

            # else:
            #     epochs_no_improve += 1
            #     print(
            #         "No improvement for %d/%d epochs. Best epoch: %d, best val loss: %.6f"
            #         % (epochs_no_improve, patience, best_epoch, best_val_loss)
            #     )

            # if epochs_no_improve >= patience:
            #     print(
            #         "Early stopping at epoch %d. Best epoch: %d, best val loss: %.6f"
            #         % (epoch, best_epoch, best_val_loss)
            #     )
            #     break
            
    elif is_for_training and is_consistency:
        print('consistency training')
        student = Seg_3DNet_2task(1, 1).to(device)
        teacher = Seg_3DNet_2task(1, 1).to(device)
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad = False
        all_subjects = sorted(glob.glob(TRAIN_DIR_PATH + '/*'))
        train_subjects, val_subjects = train_test_split(all_subjects,test_size=0.05, random_state=42, shuffle=True)
        train_dataset = TrainingDataset(train_subjects, augment=True)
        val_dataset = TrainingDataset(val_subjects, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BatchSize, shuffle=False,num_workers=WORKERSNUM,pin_memory=True) 
        cudnn.benchmark = True
        optimizer = optim.Adam(student.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # best_val_loss = float("inf")
        # best_epoch = -1

        # patience = 15
        # min_delta = 1e-4
        # epochs_no_improve = 0

        headers = {
            lossfile1: "epoch,avg_la_loss\n",
            lossfile2: "epoch,avg_scar_loss\n",
            lossfile11: "epoch,avg_sdf_loss\n",
            lossfile21: "epoch,avg_scar_mask_loss_1\n",
            lossfile22: "epoch,avg_scar_mask_loss_2\n",
            lossfile_cons: "epoch,avg_consistency_loss,consistency_weight\n",
            lossfile_la_dice: "epoch,avg_la_dice\n",
            lossfile_scar_dice: "epoch,avg_scar_dice\n",
        }
        for f, header in headers.items():
            with open(f, 'w') as file:
                file.write(header)
        for epoch in range(NumEPOCH):
            train_loss = Train_process_consistency(train_loader,student,teacher,epoch,optimizer,TRAIN_SAVE_DIR_Seg_Con)
            val_loss = Validate(val_loader, teacher, epoch)
            # if val_loss < best_val_loss - min_delta:
            #     best_val_loss = val_loss
            #     best_epoch = epoch
            #     epochs_no_improve = 0

            #     best_teacher_path = os.path.join(TRAIN_SAVE_DIR_best_CON, 'best_teacher_net.pkl')
            #     best_student_path = os.path.join(TRAIN_SAVE_DIR_best_CON, 'best_student_net.pkl')

            #     torch.save(teacher.state_dict(), best_teacher_path)
            #     torch.save(student.state_dict(), best_student_path)

            #     print("Saved best teacher/student model at epoch %d with val loss %.6f" % (epoch, val_loss))
            # else:
            #     epochs_no_improve += 1
            #     print(
            #         "No improvement for %d/%d epochs. Best epoch: %d, best val loss: %.6f"
            #         % (epochs_no_improve, patience, best_epoch, best_val_loss)
            #     )

            # if epochs_no_improve >= patience:
            #     print(
            #         "Early stopping at epoch %d. Best epoch: %d, best val loss: %.6f"
            #         % (epoch, best_epoch, best_val_loss)
            #     )
            #     break
                
    else:
        str_for_action = 'testing'
        print(str_for_action + ' .... ')
        net = Seg_3DNet_2task(1, 1).to(device)
        checkpoint_dir = TRAIN_SAVE_DIR_Seg_Con if is_consistency else TRAIN_SAVE_DIR_Seg
        checkpoint_prefix = 'teacher_net_with_' if is_consistency else 'net_with_'
        Seg_net_param = _find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
        print('Loading checkpoint:', Seg_net_param)
        F_LoadParam(Seg_net_param, net)
        net.eval()

        #! remember to change here
        # datafile = glob.glob(TEST_DIR_PATH + '/*')
        datafile = glob.glob(TRAIN_DIR_PATH + '/*')
        total_la_dice = 0.0
        total_scar_dice = 0.0
        total_subjects = 0

        for subjectid in range(len(datafile)):
            imagename = datafile[subjectid] + '/enhanced.nii.gz'
            LAlabelname = datafile[subjectid] + '/atriumSegImgMO.nii.gz'
            LAscarMaplabelname =datafile[subjectid] + '/scarSegImgM.nii.gz'

            numpyimage, numpylabel_LA, _, _, numpyprob_scar = LoadDataset_scar(
                imagename,
                LAlabelname,
                LAscarMaplabelname,
                augment=False
            )
            tensorimage = torch.from_numpy(numpyimage).unsqueeze(0).float().to(device)
            tensorlabel_LA = torch.from_numpy(numpylabel_LA).unsqueeze(0).float().to(device)
            tensorprob_scar = torch.from_numpy(numpyprob_scar).unsqueeze(0).float().to(device)

            with torch.no_grad():
                out_la, out_scar = net(tensorimage)

            la_dice = binary_dice_score(out_la, tensorlabel_LA).mean().item()
            scar_dice = binary_dice_score(out_scar[:, 1:2], tensorprob_scar).mean().item()
            total_la_dice += la_dice
            total_scar_dice += scar_dice
            total_subjects += 1
            print(
                'Test subject %d/%d, LA Dice: %.6f, Scar Dice: %.6f, path: %s'
                % (subjectid + 1, len(datafile), la_dice, scar_dice, datafile[subjectid])
            )

            predict_LA, predict_scar = ProcessTestDataset(imagename, LAlabelname, LAscarMaplabelname, net)
            # savedir = datafile[subjectid].replace('test_data', 'test_data_result_Gson')
            savedir = datafile[subjectid].replace('train_data', 'train_data_result_Gson')
            # savedir = datafile[subjectid].replace('Data_' + fold_name + '/test_data', 'fold_result')
            F_mkdir(savedir)
            nib.save(predict_LA, savedir + '/LA_predict_AJSQnet_SESA.nii.gz')
            nib.save(predict_scar, savedir + '/scar_predict_AJSQnet_SESA.nii.gz')

        if total_subjects > 0:
            print(
                'Test summary on cropped tensors, mean LA Dice: %.6f, mean Scar Dice: %.6f'
                % (total_la_dice / total_subjects, total_scar_dice / total_subjects)
            )
        print(str_for_action + ' end ')

if __name__ == '__main__':
    main()
