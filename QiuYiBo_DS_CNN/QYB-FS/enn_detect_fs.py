from __future__ import division, absolute_import, print_function
import argparse

from common.util import *
from setup_paths import *
os.environ["NPY_NUM_ARRAY_FUNCTION_ARGUMENTS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fs.datasets.datasets_utils import *
from fs.utils.squeeze import *
from fs.utils.output import write_to_csv
from fs.robustness import evaluate_robustness
from fs.detections.base import DetectionEvaluator, evalulate_detection_test, get_tpr_fpr
from DS_Murphy import *
import pandas as pd

# from tensorflow.python.platform import flags
# FLAGS = flags.FLAGS
# flags.DEFINE_boolean('detection_train_test_mode', True, 'Split into train/test datasets.')
def get_k(model, dataset, X1,act_set,mode):
    #print("ECNN预测样本，十一个概率值")
    X1_pred = model.predict(X1)
    #10是num_classes
    #print(X1_pred)
    resutls = tf.argmax(X1_pred,-1)
    #print(resutls)
    X1_pred_dict=change_to_dict(X1_pred,10)
    #print(X1_pred_dict)
    k=[]
    if dataset == 'mnist':
        if mode=="bit_depth_1":
            X1_seqeezed_bit = bit_depth_py(X1, 1)  # 位深度压缩~四舍五入
            X1_pred_seqeezed_bit=model.predict(X1_seqeezed_bit)
            #print("ECNN预测位压缩样本，十一个概率值")
            #print(X1_pred_seqeezed_bit)
            X1_pred_seqeezed_bit_dict=change_to_dict(X1_pred_seqeezed_bit,10)
            #print(X1_pred_seqeezed_bit_dict)
            k_seqeezed_bit=[]
            for n in range(X1_pred.shape[0]):
                k_seqeezed_bit.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_bit_dict[n]))
            #print("K")
            #print(k_seqeezed_bit)
            k.append(k_seqeezed_bit)
        elif mode=="bit_depth_2":
            X1_seqeezed_bit = bit_depth_py(X1, 2)  # 位深度压缩~四舍五入
            X1_pred_seqeezed_bit=model.predict(X1_seqeezed_bit)
            #print("ECNN预测位压缩样本，十一个概率值")
            #print(X1_pred_seqeezed_bit)
            X1_pred_seqeezed_bit_dict=change_to_dict(X1_pred_seqeezed_bit,10)
            #print(X1_pred_seqeezed_bit_dict)
            k_seqeezed_bit=[]
            for n in range(X1_pred.shape[0]):
                k_seqeezed_bit.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_bit_dict[n]))
            #print("K")
            #print(k_seqeezed_bit)
            k.append(k_seqeezed_bit)
        elif mode=="median_filter_2":
            X1_seqeezed_filter_median = median_filter_py(X1, 2)  # 中值滤波
            X1_pred_seqeezed_filter_median=model.predict(X1_seqeezed_filter_median)
            #print("ECNN预测中值滤波压缩样本，十一个概率值")
            #print(X1_pred_seqeezed_filter_median)
            X1_pred_seqeezed_filter_median_dict=change_to_dict(X1_pred_seqeezed_filter_median,10)
            #print(X1_pred_seqeezed_filter_median_dict)
            k_seqeezed_filter_median=[]
            for n in range(X1_pred.shape[0]):
                k_seqeezed_filter_median.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_filter_median_dict[n]))
            #print("K")
            #print(k_seqeezed_filter_median)
            k.append(k_seqeezed_filter_median)
        elif mode=="median_filter_3":
            X1_seqeezed_filter_median = median_filter_py(X1, 3)  # 中值滤波
            X1_pred_seqeezed_filter_median=model.predict(X1_seqeezed_filter_median)
            #print("ECNN预测中值滤波压缩样本，十一个概率值")
            #print(X1_pred_seqeezed_filter_median)
            X1_pred_seqeezed_filter_median_dict=change_to_dict(X1_pred_seqeezed_filter_median,10)
            #print(X1_pred_seqeezed_filter_median_dict)
            k_seqeezed_filter_median=[]
            for n in range(X1_pred.shape[0]):
                k_seqeezed_filter_median.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_filter_median_dict[n]))
            #print("K")
            #print(k_seqeezed_filter_median)
            k.append(k_seqeezed_filter_median)
        elif mode=="bit_depth_1&median_filter_2":
            X1_seqeezed_bit = bit_depth_py(X1, 1)  # 位深度压缩~四舍五入
            X1_pred_seqeezed_bit=model.predict(X1_seqeezed_bit)
            #print("ECNN预测位压缩样本，十一个概率值")
            #print(X1_pred_seqeezed_bit)
            X1_pred_seqeezed_bit_dict=change_to_dict(X1_pred_seqeezed_bit,10)
            #print(X1_pred_seqeezed_bit_dict)
            k_seqeezed_bit=[]
            for n in range(X1_pred.shape[0]):
                k_seqeezed_bit.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_bit_dict[n]))
            #print("K")
            #print(k_seqeezed_bit)
            k.append(k_seqeezed_bit)
            X1_seqeezed_filter_median = median_filter_py(X1, 2)  # 中值滤波
            X1_pred_seqeezed_filter_median=model.predict(X1_seqeezed_filter_median)
            #print("ECNN预测中值滤波压缩样本，十一个概率值")
            #print(X1_pred_seqeezed_filter_median)
            X1_pred_seqeezed_filter_median_dict=change_to_dict(X1_pred_seqeezed_filter_median,10)
            #print(X1_pred_seqeezed_filter_median_dict)
            k_seqeezed_filter_median=[]
            for n in range(X1_pred.shape[0]):
                k_seqeezed_filter_median.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_filter_median_dict[n]))
            #print("K")
            #print(k_seqeezed_filter_median)
            k.append(k_seqeezed_filter_median)

    else:
        X1_seqeezed_bit = bit_depth_py(X1, 5)
        X1_pred_seqeezed_bit=model.predict(X1_seqeezed_bit)
        #print("ECNN预测位压缩样本，十一个概率值")
        #print(X1_pred_seqeezed_bit)
        X1_pred_seqeezed_bit_dict=change_to_dict(X1_pred_seqeezed_bit,10)
        #print(X1_pred_seqeezed_bit_dict)
        k_seqeezed_bit=[]
        for n in range(X1_pred.shape[0]):
            k_seqeezed_bit.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_bit_dict[n]))
        #print("K")
        #print(k_seqeezed_bit)
        k.append(k_seqeezed_bit)
        X1_seqeezed_filter_median = median_filter_py(X1, 2)
        X1_pred_seqeezed_filter_median=model.predict(X1_seqeezed_filter_median)
        #print("ECNN预测中值滤波压缩样本，十一个概率值")
        #print(X1_pred_seqeezed_filter_median)
        X1_pred_seqeezed_filter_median_dict=change_to_dict(X1_pred_seqeezed_filter_median,10)
        #print(X1_pred_seqeezed_filter_median_dict)
        k_seqeezed_filter_median=[]
        for n in range(X1_pred.shape[0]):
           k_seqeezed_filter_median.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_filter_median_dict[n]))
        #print("K")
        #print(k_seqeezed_filter_median)
        k.append(k_seqeezed_filter_median)
        X1_seqeezed_filter_local = non_local_means_color_py(X1, 13, 3, 2)
        X1_pred_seqeezed_filter_local=model.predict(X1_seqeezed_filter_local)
        #print("ECNN预测非局部滤波压缩样本，十一个概率值")
        #print(X1_pred_seqeezed_filter_local)
        X1_pred_seqeezed_filter_local_dict=change_to_dict(X1_pred_seqeezed_filter_local,10)
        #print(X1_pred_seqeezed_filter_local_dict)
        k_seqeezed_filter_local=[]
        for n in range(X1_pred.shape[0]):
           k_seqeezed_filter_local.append(computrEmpty(X1_pred_dict[n],X1_pred_seqeezed_filter_local_dict[n]))
        #print("K")
        #print(k_seqeezed_filter_local)
        k.append(k_seqeezed_filter_local)
    return np.max(k, axis=0)

def train_fs(model, dataset, X1, train_fpr,act_set,mode):
    k = get_k(model, dataset, X1,act_set,mode)
    selected_distance_idx = int(np.ceil(len(X1) * (1 - train_fpr)))
    threshold = sorted(k)[selected_distance_idx - 1]
    threshold = threshold
    #选择与预测值之间距离阈值
    print("Threshold value: %f" % threshold)
    return threshold

#将距离大于阈值的样本标记为正类（True），将距离小于或等于阈值的样本标记为负类（False）
def test(model, dataset, X, threshold,act_set,mode):
    k = get_k(model, dataset, X,act_set,mode)
    Y_pred = k > threshold
    return Y_pred, k

def compare_uncertainty(model, dataset,x_le,x_adv):
    y_le_pre=model.predict(x_le)
    y_adv_pre=model.predict(x_adv)
    y_le_uncer_pre=y_le_pre[:, 10]*1e5
    y_adv_uncer_pre=y_adv_pre[:, 10]*1e5
    return y_le_uncer_pre,y_adv_uncer_pre



def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]

    assert os.path.isfile('{}cnn_{}.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model using train_model.py.'

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'mnist':
        from baselineCNN.ecnn.ecnn_mnist import MNISTECNN as myECNNModel
        from baselineCNN.cnn.cnn_mnist import MNISTCNN as myCNNModel
        model_class_ecnn = myECNNModel(mode='load', filename='ecnn_{}.h5'.format(args.dataset))
        model_class_cnn = myCNNModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model_ecnn = model_class_ecnn.model
        model_cnn = model_class_cnn.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model_ecnn.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        model_cnn.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])


    elif args.dataset == 'cifar':
        from baselineCNN.ecnn.enn_cifar10 import CIFAR10ENN as myENNModel
        from baselineCNN.ecnn.cnn_cifar10 import CIFAR10CNN as myCNNModel
        model_class_ecnn = myENNModel(mode='load', filename='new_ecnn_{}.h5'.format(args.dataset))
        model_class_cnn = myCNNModel(mode='load', filename='new_cnn_{}.h5'.format(args.dataset))
        model_ecnn = model_class_ecnn.model
        model_cnn = model_class_cnn.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_ecnn.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        model_cnn.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        # model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), 
        #       loss='CategoricalCrossentropy',
        #       metrics=['accuracy'])

    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model = model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model = model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    # Load the dataset
    X_train_all, Y_train_all, X_test_all, Y_test_all = model_class_cnn.x_train, model_class_cnn.y_train, model_class_cnn.x_test, model_class_cnn.y_test
    act_set=model_class_ecnn.act_set
    act_set = np.array(act_set)
    #print("x_train_all")
    #print(X_train_all[:5])
    # --------------
    # Evaluate the trained model.
    # Refine the normal and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    #用cnn抽取正确数据集
    #print(act_set)
    print("Evaluating the pre-trained model...")
    Y_pred_all = model_ecnn.predict(X_test_all)
    # print(Y_pred_all.shape)
    # print(Y_pred_all[:5])
    # print(Y_test_all[:5])
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    #代码从预测结果中筛选出原始版本被正确分类的样本
    # 并将这些样本存储在X_test和Y_test数组中
    # 而且也存储它们的预测结果在Y_pred数组中
    inds_correct = np.where(Y_pred_all.argmax(axis=1) == Y_test_all.argmax(axis=1))[0]
    X_test = X_test_all[inds_correct]
    Y_test = Y_test_all[inds_correct]
    Y_pred = Y_pred_all[inds_correct]
    #random.sample函数从X_test和Y_test中随机选择一些样本作为特征选择器的训练集
    #indx_train = random.sample(range(len(X_test)), int(len(X_test) / 2))
    indx_train = random.sample(range(len(X_test)), int(len(X_test) / 2))
    indx_test = list(set(range(0, len(X_test))) - set(indx_train))
    #list(set())函数将这些样本从测试集中移除
    #indx_test=indx_train
    #indx_test=indx_train
    print("Number of correctly predict images: %s" % (len(inds_correct)))
    x_train = X_test[indx_train]
    y_train = Y_test[indx_train]
    x_test = X_test[indx_test]
    y_test = Y_test[indx_test]
    #print(x_train)
    # compute thresold - use test data to compute that
    print("cnn预测普通样本")
    x_train_pre=model_cnn.predict(x_train)
    train_classifier_indices=tf.argmax(x_train_pre,axis=1)
    print(train_classifier_indices)
    #求出K
    #get_k(model_ecnn, args.dataset, x_train,act_set)
    Y_test_copy = Y_test
    X_test_copy = X_test
    y_test_copy = y_test
    x_test_copy = x_test
    ## Evaluate detector
    # on adversarial attack
    # Create empty dataframe
    #df = pd.DataFrame(columns=['Attack', 'Threshold', 'Mode', 'Accuracy'])
    for attack in ATTACKS:
        df_all = pd.DataFrame()
        df_success = pd.DataFrame()
        df_fail = pd.DataFrame()
        Y_test = Y_test_copy
        X_test = X_test_copy
        y_test = y_test_copy
        x_test = x_test_copy
        results_all = []

        # Prepare data
        # Load adversarial samples
        #改成了fs原作者中的攻击样本
        #X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_dir, args.dataset, attack))
        X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_dir, args.dataset, attack))

        #reduce_precision_py将图像的精度降低
        X_test_adv = reduce_precision_py(X_test_adv, 256)
        #根据不同的攻击类型和数据集来选择测试数据
        if attack == 'df' and args.dataset == 'tiny':
            Y_test = model_class.y_test[0:2700]
            X_test = model_class.x_test[0:2700]
            cwi_inds = inds_correct[inds_correct < 2700]
            Y_test = Y_test[cwi_inds]
            X_test = X_test[cwi_inds]
            X_test_adv = X_test_adv[cwi_inds]
            xtest_inds = np.asarray(indx_test)[np.asarray(indx_test) < 2700]
            xtest_inds = np.in1d(cwi_inds, xtest_inds)
            x_test = X_test[xtest_inds]
            y_test = Y_test[xtest_inds]
            X_test_adv = X_test_adv[xtest_inds]
        else:
            X_test_adv = X_test_adv[inds_correct]
            X_test_adv = X_test_adv[indx_test]
        #评估模型在对抗样本上的性能，并将成功和失败的样本分别存储在不同的数组中
        loss, acc_suc = model_cnn.evaluate(X_test_adv, y_test, verbose=0)
        print("判断在%s被攻击的样本上的情况"%attack)
        print("ecnn预测攻击样本")
        X_test_adv_pred = model_ecnn.predict(X_test_adv)
        # print(X_test_adv_pred)
        test_classifier_indices=tf.argmax(X_test_adv_pred,axis=1)
        print(test_classifier_indices)
        print(acc_suc)


        #判断第十一项的变化情况
        # y_le_uncer_pre,y_adv_uncer_pre=compare_uncertainty(model_ecnn, args.dataset,x_test,X_test_adv)
        # print("%s"%attack)
        # print(y_le_uncer_pre)
        # print(y_adv_uncer_pre)
        # shape = x_test.shape
        # # 获取y_le_pre[:, 10]的长度
        # length = shape[0]
        # # 输出结果
        # print("y_le_pre[:, 10]一共有", length, "项")
        # # 统计y_le_pre[:, 10]中有多少项大于y_adv_pre[:, 10]
        # count = np.sum(y_le_uncer_pre < y_adv_uncer_pre)

        # # 输出结果
        # print("有", count, "项小于y_adv_pre[:, 10]")

        #攻击成功，错误识别
        inds_success = np.where(X_test_adv_pred.argmax(axis=1) != y_test.argmax(axis=1))[0]
        #攻击失败，正确识别
        inds_fail = np.where(X_test_adv_pred.argmax(axis=1) == y_test.argmax(axis=1))[0]
        inds_all_not_fail = list(set(range(0, len(inds_correct)))-set(inds_fail))
        X_test_adv_success = X_test_adv[inds_success]
        Y_test_success = y_test[inds_success]
        X_test_adv_fail = X_test_adv[inds_fail]
        Y_test_fail = y_test[inds_fail]
        #准备用于检测器的数据集，将原始测试集和对抗样本合并，并将它们标记为成功分类或失败分类
        # prepare X and Y for detectors
        X_all = np.concatenate([x_test, X_test_adv])
        Y_all = np.concatenate([np.zeros(len(x_test), dtype=bool), np.ones(len(x_test), dtype=bool)])
        X_success = np.concatenate([x_test[inds_success], X_test_adv_success])
        Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
        X_fail = np.concatenate([x_test[inds_fail], X_test_adv_fail])
        Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

        modes=["bit_depth_1","bit_depth_2","median_filter_2","median_filter_3","bit_depth_1&median_filter_2"]
        for mode in modes:
            print("%s压缩器"%mode)
            threshold = train_fs(model_ecnn, args.dataset, x_train, 0.05,act_set,mode)
            Y_all_pred, Y_all_pred_score = test(model_ecnn, args.dataset, X_all, threshold,act_set,mode)
            Y_success_pred, Y_success_pred_score = test(model_ecnn, args.dataset, X_success, threshold,act_set,mode)
            Y_fail_pred, Y_fail_pred_score = test(model_ecnn, args.dataset, X_fail, threshold,act_set,mode)
            #print(Y_all_pred)
            #print(Y_all_pred_score)
            acc_all = accuracy_score(Y_all, Y_all_pred)
            acc_success = accuracy_score(Y_success, Y_success_pred)
            acc_fail = accuracy_score(Y_fail, Y_fail_pred)
            print("检测率_all")
            print(acc_all)
            print("检测率_sucess")
            print(acc_success)
            print("检测率_fail")
            print(acc_fail)
            # Add data to dataframe
            new_row_all = pd.DataFrame({'Attack': attack, 'Threshold': threshold, 'Mode': mode, 'Accuracy': acc_all}, index=[0])
            new_row_success = pd.DataFrame({'Attack': attack, 'Threshold': threshold, 'Mode': mode, 'Accuracy': acc_success}, index=[0])
            new_row_fail = pd.DataFrame({'Attack': attack, 'Threshold': threshold, 'Mode': mode, 'Accuracy': acc_fail}, index=[0])
            # Concatenate new dataframe with existing dataframe
            df_all = pd.concat([df_all, new_row_all], ignore_index=True)
            df_success = pd.concat([df_success, new_row_success], ignore_index=True)
            df_fail = pd.concat([df_fail, new_row_fail], ignore_index=True)
        # Save dataframe to CSV file
        df_all.to_csv('mnist_results_all.csv', index=False, mode='a', header=False)
        df_success.to_csv('mnist_results_success.csv', index=False, mode='a', header=False)
        df_fail.to_csv('mnist_results_fail.csv', index=False, mode='a', header=False)


        # for Y_all
        # if attack == ATTACKS[0]:
        #测试并评估检测器在整个数据集上的性能
        #Y_all_pred, Y_all_pred_score = test(model_ecnn, args.dataset, X_all, threshold,act_set)
        #get_k(model_ecnn, args.dataset, X_test_adv, act_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either {}".format(DATASETS),
        required=True, type=str
    )
    args = parser.parse_args()
    main(args)