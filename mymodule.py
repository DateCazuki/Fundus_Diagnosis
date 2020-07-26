import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np

class BatchGenerator(keras.utils.Sequence):
    def __init__(self,image_path,label,img_shape,batch_size):
        self.x = image_path
        self.y = label
        self.length = len(image_path)

        self.batch_size = batch_size
        self.image_shape = img_shape
        self.batches_per_epoch = int((self.length-1)/batch_size) + 1
        #int(X/Y)+1はX/Yの切り上げ処理．ただしmod(X/Y)=0のときのために，X-1の処理を行う．

    def __getitem__(self,idx):
    #__getitem__による定義：
    #   Python特殊メソッドの一つ．オブジェクトに角括弧'[]'でアクセスしたときの挙動を定義．
    #  __getitem__内ではxは順次読み出しされるため，事前にランダム化されていなければならない．

        batch_from = self.batch_size * idx
        batch_to   = batch_from+self.batch_size
        if batch_to > self.length:
            batch_to = self.length

        x_batch = []
        y_batch = []
#        print("idx : {}".format(idx))
        for i in range(batch_from,batch_to):
            img = load_img(self.x[i], color_mode="rgb")
            img = img_to_array(img)/255.0
            x_batch.append(img)
            y_batch.append(self.y[i])

        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)

        x_batch = x_batch.reshape(x_batch.shape[0],self.image_shape[0],
                                    self.image_shape[1],self.image_shape[2])

#        return x_batch,y_batch
        return x_batch,y_batch


    def __len__(self):
    ##__len__による定義：
    #  クラスが値の集合のためのコンテナとしてふるまう場合に，
    #　そのクラスのシーケンスの長さを呼び出すための定義
    #　len(インスタンス名)での帰り値を定義する．
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass

def Make_Raw_List(img_folder_path,csv_filename):
    '''Make_Raw_List(img_folder_path,csv_file):
    <Input>
    　img_folder_path:画像ファイルの入ったフォルダへのパス
    　csv_file:csv_fileへのパス＋ファイル名
    <Output>
    　img_path_array:画像ファイル名 ndarray
      teacher_array:教師データndarray(one-hot, float32)
    　
    元データとなるcsvファイルに対して，下記事前処理をおこなう．
    　－　各画像ファイル名にパスを追加
    　－　分類先該当なしの場合に，one-hotベクトルの最終段に'1'を追加．
    　－　不要なデータを削除(ここでは年齢，性別など)

    '''
    tmp_img_list   = []
    tmp_teacher_list   = []

    with open(csv_filename,'r') as f:
        next(f)                         #ファイルの1行目を飛ばす
        for line in f:
            line = line.rstrip('\n')    #改行コードの削除
            data = line.split(',')      #カンマごとに分割

            #labelデータ用の二次元リスト生成
            # 　該当項目がない場合，つまりone-hotベクトルがすべて'0'の場合，
            # 　最後に'1'を追加する．
            if '1' in data[5:]:
                data.append('0')
            else:
                data.append('1')

            data[0] = img_folder_path + '/' + data[0]

            tmp_img_list.append(data[0])
            tmp_teacher_list.append(data[5:])

    img_path_array = np.array(tmp_img_list)
    teacher_array  = np.array(tmp_teacher_list,dtype=np.float32)

#    return img_path_array, teacher_array
    return img_path_array, teacher_array

def Make_Raw_List2(img_folder_path,csv_filename):
    '''Make_Raw_List(img_folder_path,csv_file):
    <Input>
    　img_folder_path:画像ファイルの入ったフォルダへのパス
    　csv_file:csv_fileへのパス＋ファイル名
    <Output>
    　img_path_array:画像ファイル名 ndarray
      teacher_array:教師データndarray(one-hot, float32)
    　
    元データとなるcsvファイルに対して，下記事前処理をおこなう．
    　－　各画像ファイル名にパスを追加
      －　複数“1”の立っている項目の削除
      －　DR(data[9])の削除
      －　AO(data[12])の削除
    　－　分類先該当なしの場合に，one-hotベクトルの最終段に'1'を追加．
    　－　不要なデータを削除(ここでは年齢，性別など)

    '''
    IMG_FILE = 0
    LABEL_START = 5
    DR = 9
    AO = 12

    tmp_img_list   = []
    tmp_teacher_list   = []

    with open(csv_filename,'r') as f:
        next(f)                         #ファイルの1行目を飛ばす
        for line in f:

            line = line.rstrip('\n')    #改行コードの削除
            data = line.split(',')      #カンマごとに分割

            #複合症例(複数のラベルが該当するケース)は削除
            if data[LABEL_START:].count('1') > 1:
                continue
            elif data[DR] == '1':
                continue
            elif data[AO] == '1':
                continue

            #labelデータ用の二次元リスト生成
            # 　該当項目がない場合，つまりone-hotベクトルがすべて'0'の場合，
            # 　最後に'1'を追加する．
            if '1' in data[LABEL_START:]:
                data.append('0')
            else:
                data.append('1')

            data[IMG_FILE] = img_folder_path + '/' + data[0]

            tmp_img_list.append(data[IMG_FILE])
            tmp_teacher_list.append(data[LABEL_START:])
            DR_LS = DR - LABEL_START
            AO_LS = AO - LABEL_START

    img_path_array = np.array(tmp_img_list)
    teacher_array  = np.array(tmp_teacher_list,dtype=np.float32)
    print(teacher_array)
    teacher_array=np.delete(teacher_array,[DR_LS,AO_LS],axis=1)

#    return img_path_array, teacher_array
    return img_path_array, teacher_array


def plot_loss_accuracy_graph(fit_record):
    fig = plt.figure(figsize=(15,5))
    loss_graph = fig.add_subplot(121,title='LOSS',xlabel='Epochs', ylabel='Loss')
    acc_graph = fig.add_subplot(122,title='ACCURACY',xlabel='Epochs', ylabel='Accuracy')

    loss_graph.plot(fit_record.history['loss'],'-D',color='blue',label='train_loss',linewidth=2)
    loss_graph.plot(fit_record.history['val_loss'],'-D',color='black',label='val_loss',linewidth=2)
    loss_graph.legend(loc='upper right')

    acc_graph.plot(fit_record.history['accuracy'],'-o',color='green',label='train_accuracy',linewidth=2)
    acc_graph.plot(fit_record.history['val_accuracy'],'-o',color='black',label='valu_accyracy',linewidth=2)
    acc_graph.legend(loc='lower right')

    plt.show()
