import matplotlib.pyplot as plt
import numpy as np
# Hyper parameters
# HPM or withoutHPM
isHPM = 'HPM'
model_type = 'VGG16'
model_type1 = 'VGG16'
model_type2 = 'ResNet50'
load_epoch = 49
batch_size = 16
loss_weight = 10.0

# TestData Parameters
AreaType = "liangxiang"
SatLevel = "18"
GrdType = "notAligned"
train_data_type = 'CVACT'
test_data_type = AreaType + "_" + SatLevel + "_" + GrdType

def draw_compare():
    print('loading multiple dist_array and recall_accuracy...')

    dist_array1 = np.load(
        '../Visualize/' + train_data_type + '/' + model_type1 + '/' + test_data_type + "/"  + isHPM + '/dist_array.npy')
    val_accuracy1 = np.load(
        '../Visualize/' + train_data_type + '/' + model_type1 + '/' + test_data_type + "/"  + isHPM + '/recall_accuracy.npy')
    val_accuracy2 = np.load(
        '../Visualize/' + train_data_type + '/' + model_type2 + '/' + test_data_type + "/"  + isHPM + '/recall_accuracy.npy')

    print('loaded')

    # top1_percent = int(dist_array1.shape[0] * 0.01) + 1
    top1_percent = int(dist_array1.shape[0])
    k = np.arange(top1_percent)
    plt.plot(k, val_accuracy1[0] * 100, label=model_type1)
    plt.plot(k, val_accuracy2[0] * 100, label=model_type2)
    plt.title('Comparision' + "_Trained_on_" + train_data_type + "_Tested_on_" + test_data_type + "_" + isHPM)
    plt.xlabel(r'k')
    plt.ylabel(r"recall rate at @ top-k")

    plt.legend()
    plt.show()
    print('comparision_recall@top-K has been saved')

if __name__ == '__main__':
    draw_compare()