import torch
import config
from torch.utils.data.dataloader import DataLoader
from model import mytEEGTransformer
from dataloader import Mydataset, get_train_test_clip
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from openAI import run
import msvcrt
scaler = StandardScaler()
BATCHSIZE = 2
device = config.device

def LOCO(sub, clip_num, learning_rate, batch_size, epoch):

    with open(config.save_index_1 + '/result_LOCO_dep/result_' + sub + '_' + 'clip' + str(clip_num) + '.txt', 'w') as f:

        x_train, y_train, x_test, y_test = get_train_test_clip(sub, clip_num)
        print('subject: ', sub)
        print('trial_num: ', clip_num)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        test_ds = Mydataset(x_test[:, :config.window_size], x_test[:, -1], y_test)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=True)

        net = mytEEGTransformer(6, config.channels_num, 4, 1)
        net.to(device)

        print('Loading weights into state dict ...')
        pretrained_params = torch.load(config.save_index_1 + '/canshu/trained_classfication_' + sub + '_' + 'clip' + str(clip_num)+ '.pt')
        net.load_state_dict(pretrained_params, strict=False)
        net.eval()
        with torch.no_grad():

            for i, data in enumerate(test_loader):
                in_shots, _, label = data
                in_shots = in_shots.permute([0, 1, 3, 2])
                in_shots, label = in_shots.to(device), label.to(device)
                predicted_shot_two, _ = net(in_shots)
                predictions = torch.argmax(predicted_shot_two, dim=1)
                probabilities = F.softmax(predicted_shot_two, dim=-1)

                print("pre:", predictions[0].item())
                print("tru:", label[0].item())
                if predictions[0].item() == 0:     #can be self-defined
                    user_message = "Please create a cheerful verse about happiness and celebration,not exceeding around 40 words"
                elif predictions[0].item() == 1:
                    user_message = "Please tell a joke "
                elif predictions[0].item() == 2:
                    user_message = "Say some passionate words, not exceeding around 40 words."
                else:
                    user_message = "Say something to overcome fear,not exceeding around 40 words"
                run(predictions[0].item(), user_message)

    return None

sub = '1'
for clip_num in config.clip_num_list:
##分类
    with open(config.save_index_1 + '/result_LOCO_dep/result_' + sub + '_' + 'clip' + str(clip_num) + '.txt', 'w') as f:
        print('The classfication result', file=f)
        LOCO(sub, clip_num, learning_rate=config.lr_classfication_optimizer, batch_size=BATCHSIZE, epoch=config.classification_epoch)
        f.close()
        input("继续")



