import os
import sys
import torch
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from SPAnet import spa_tiny_224 as create_model
from utils import MyDataSet, read_test_data
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
import seaborn as sns
import json

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    print(args)

    test_images_path, test_images_label = read_test_data(args.data_path)

    num_classes = args.num_classes
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(256)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = args.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # calculation and evaluation index
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    labels = []
    pred_classes = []
    data_loader = tqdm(test_loader, file=sys.stdout)

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            imgs, label = data
            sample_num += imgs.shape[0]

            pred = model(imgs.to(device))
            pred_class = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_class, label.to(device)).sum()

            label = label.cpu().detach().numpy()
            pred_class = pred.cpu().detach().numpy()
            labels.extend(label)
            pred_classes.extend(pred_class)
    # ACC
    accuracy = accu_num.item()/ sample_num

    # AUC
    labels = np.array(labels)
    pred_classes = np.array(pred_classes)
    n_classes = pred_classes.shape[1]
    aucs = []
    fprs = []
    tprs = []
    for i in range(n_classes):
        fpr, tpr, thresholds = metrics.roc_curve(labels, pred_classes[:, i], pos_label=i)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
        fprs.append(fpr)
        tprs.append(tpr)
    macro_auc = np.mean(aucs)

    json_file = open('class_indices.json', 'r')
    class_indices = json.load(json_file)

    cls_name_all = []
    # Plot ROC curves for each class
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(n_classes):
        cls_name = class_indices[str(i)]
        cls_name_all.append(cls_name)
        plt.plot(fprs[i], tprs[i], label='{} (AUC = {:.2f})'.format(cls_name, aucs[i]))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curves')
    plt.legend(loc='lower right')
    plt.show()

    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(labels, pred_classes.argmax(axis=1))
    normalized_confusion_matrix = np.around(
        confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=4)
    plt.figure(figsize=(13, 11))
    sns.heatmap(normalized_confusion_matrix, annot=True, cmap='Blues', cbar=True, fmt=".4f")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    # Change the tick labels
    # class_name_all = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus',
    #                'normal-z-line', 'polyps', 'ulcerative-colitis']
    plt.xticks(np.arange(len(cls_name_all)) + 0.5, cls_name_all, rotation=45)
    plt.yticks(np.arange(len(cls_name_all)) + 0.5, cls_name_all, rotation=0)
    plt.show()

    # acc, precision, recall, f1
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, pred_classes.argmax(axis=1),
                                                                       average='macro', zero_division=False)
    print("acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(accuracy, precision, recall, f1))
    print(normalized_confusion_matrix)

    # MCC, Kappa, AUC
    mcc = matthews_corrcoef(labels, pred_classes.argmax(axis=1))
    kappa = cohen_kappa_score(labels, pred_classes.argmax(axis=1))
    print("MCC: {:.4f}, Kappa: {:.4f}, AUC: {:.4f}".format(mcc, kappa, macro_auc))

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-path', type=str, default="data/kvasirv2/test")
    parser.add_argument('--weights', type=str, default='model_weight/best_model.pth', help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)