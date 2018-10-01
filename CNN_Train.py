import torch.optim as optim
import torch.nn
import time
from DoodleRecognition.CNN import *
from DoodleRecognition.CyclicLR import *

criterion = nn.CrossEntropyLoss()
lr = 0.001
weight_decay = 5e-4
gamma=0.1
stepsize=60
epochs = 40
num_classes=len(classes)
feature_extract = False

## use ResNet18 as pre-trained model
ft_model = torchvision.models.resnet18(pretrained=True)
model = Net(ft_model, feature_extract)
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad:
            print("\t",name)

opt = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
scheduler = CyclicLR(opt,gamma=gamma,step_size=stepsize)


def train_model(model, dataset_sizes, criterion, optimizer, scheduler, use_gpu=True, num_epochs=25, mixup = False, alpha = 0.1):
    print("MIXUP".format(mixup))
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            dataloaders = train_gen(phase)
            if phase == 'train':
                scheduler.batch_step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #for data in tqdm(dataloaders):
            for i, data in enumerate(dataloaders, 0):
                # get the inputs
                inputs, labels = data
                #augementation using mixup
                #if phase == 'train' and mixup:
                #    inputs = mixup_batch(inputs, alpha)
                # wrap them in Variable

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = model(inputs)

                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)


                if phase == 'train':
                    print('index: %s  running_loss %.4f  running_corrects %.4f' % (
                    i, running_loss / (i + 1), running_corrects.item() / ((i + 1) * batch_size)))

            epoch_loss = running_loss / (dataset_sizes[phase]/batch_size)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model=model,
                       scheduler=scheduler,dataset_sizes={'train': num_classes * n_rows, 'val': int(num_classes * n_rows * 0.1)},
                       criterion=criterion, optimizer=opt)