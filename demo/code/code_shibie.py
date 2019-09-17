import numpy as np
import os
from PIL import Image
import cv2

from sklearn.svm import SVC
from sklearn.externals import joblib
def initTable(threshold=100):           # 二值化函数
    table = []
    for i in range(256):
        if i > threshold:
            table.append(0)
        else:
            table.append(1)

    return table
def print_bin(img):
    """
    输出二值后的图片到控制台，方便调试的函数
    :param img:
    :type img: Image
    :return:
    """
    print('current binary output,width:%s-height:%s\n')
    for h in range(img.height):
        for w in range(img.width):
            value =img.getpixel((w,h))
            # print(value, end='')
            if value==0:
                print(value, end='')
            else:
                print(' ',end='')


        print('')


def sum_9_region_new(img, x, y):
    '''确定噪点 '''
    cur_pixel = img.getpixel((x, y))  # 当前像素点的值
    width = img.width
    height = img.height

    if cur_pixel == 1:  # 如果当前点为白色区域,则不统计邻域值
        return 0

    # 因当前图片的四周都有黑点，所以周围的黑点可以去除
    if y < 3:  # 本例中，前两行的黑点都可以去除
        return 1
    elif y > height - 3:  # 最下面两行
        return 1
    else:  # y不在边界
        if x < 3:  # 前两列
            return 1
        elif x == width - 1:  # 右边非顶点
            return 1
        else:  # 具备9领域条件的
            sum = img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1)) \
                  + img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 9 - sum

def collect_noise_point(img):
	'''收集所有的噪点'''
	noise_point_list = []
	for x in range(img.width):
		for y in range(img.height):
			res_9 = sum_9_region_new(img, x, y)
			if (0 < res_9 < 4) and img.getpixel((x, y)) == 0:  # 找到孤立点
				pos = (x, y)
				noise_point_list.append(pos)
	return noise_point_list

def remove_noise_pixel(img, noise_point_list):
	'''根据噪点的位置信息，消除二值图片的黑点噪声'''
	for item in noise_point_list:
		img.putpixel((item[0], item[1]), 1)





def noise_reduction(img):
    """


    :param img: 灰度处理 二值化 后的图片
    :return:
    """
    noise_point_list = collect_noise_point(img)#收集噪点
    remove_noise_pixel(img, noise_point_list)#去除噪点
    # print_bin(img)  # 输出二值图像
    return img

def smartSliceImg(img,count=4, p_w=3):
    '''
    :param img:
    :param outDir:
    :param count: 图片中有多少个图片
    :param p_w: 对切割地方多少像素内进行判断
    :return:
    '''
    w, h = img.size
    pixdata = img.load()
    eachWidth = int(w / count)
    beforeX = 0
    crop_list =[]

    for i in range(count):

        allBCount = []
        nextXOri = (i + 1) * eachWidth

        for x in range(nextXOri - p_w, nextXOri + p_w):
            if x >= w:
                x = w - 1
            if x < 0:
                x = 0
            b_count = 0
            for y in range(h):
                if pixdata[x, y] == 0:
                    b_count += 1
            allBCount.append({'x_pos': x, 'count': b_count})
        sort = sorted(allBCount, key=lambda e: e.get('count'))
        nextX = sort[0]['x_pos']
        box = (beforeX, 0, nextX, h)

        # img.crop(box).save('./Cutting/{}' + str(i) + "_" + str(i) + ".png")
        beforeX = nextX
        crop_list.append(box)
    return crop_list



def get_file_name(user_dir):
    file_list = list()
    for root, dirs, files in os.walk(user_dir):
        print('共有',len(files),'个文件')
    return files

def SliceImg(path):

    # 创建文件夹存放切割后的图片
    # for key,word in chinese.items():
    #     if os.path.exists('./Cutting/{}'.format(word)):
    #         pass
    #     else:
    #         os.mkdir('./Cutting/{}'.format(word))

    file_name_list = get_file_name(path)
    for  i in file_name_list:
        Train_Pretreatment(path+'/'+i)




def Train_Pretreatment(iamge_path):

    #训练数据预处理
    # 二值化， 去噪点  ，切割

    each_name = iamge_path.split('__',)[1].split('_')# 获取标记数据
    bianhao = iamge_path.split('__',)[2]


    im = Image.open(iamge_path)  # 1.打开图片
    im = im.convert('L')# 灰度处理
    binaryImage = im.point(initTable(), '1')#二值化
    binaryImage = noise_reduction(binaryImage) #去噪点
    crop_list = smartSliceImg(binaryImage)#返回切割坐标

    for box,each in zip(crop_list,each_name):

        binaryImage.crop(box).save('./Cutting/{}/'.format(str(each)) + str(each) + "_" + str(bianhao) )

def getletter(fn):
    fnimg = cv2.imread(fn)  # 读取图像
    img = cv2.resize(fnimg, (25, 25))  # 将图像大小调整为8*8



    alltz = []
    for now_h in range(0, 25):
        xtz = []
        for now_w in range(0, 25):
            b = img[now_h, now_w, 0]
            g = img[now_h, now_w, 1]
            r = img[now_h, now_w, 2]
            btz = 255 - b
            gtz = 255 - g
            rtz = 255 - r
            if btz > 0 or gtz > 0 or rtz > 0:
                nowtz = 1
            else:
                nowtz = 0
            xtz.append(nowtz)
        alltz += xtz
    return alltz

def load_dataset():
    X = []
    y = []

    for key,i in chinese.items():

        target_path = "./Cutting/" + str(i)
        # print(target_path)
        for title in os.listdir(target_path):

            # pix = np.asarray(Image.open(os.path.join(target_path, title)).convert('L'))
            pix = getletter(os.path.join(target_path, title))
            li = list(set(pix))# 去重 去掉全白或全黑的异常数据
            if len(li) ==1:
                pass
            else:
                X.append(pix)
                y.append(i)

            # X.append(pix)
            # y.append(i)
    X = np.asarray(X)
    y = np.asarray(y)
    print(len(X),len(y))
    return X, y

def trainSVM(X,Y):

    # 使用向量机SVM进行机器学习
    letterSVM = SVC(kernel="linear", C=1).fit(X, Y)
    # 生成训练结果
    joblib.dump(letterSVM, './result/letter.pkl')

def TestSVM(iamge_path):


    # iamge_path ='./test_data/2019y07m05d_10h13m20s__18_20_15_6__122.png'


    clf = joblib.load('./result/letter.pkl')
    each_name = iamge_path.split('__', )[1]  # 获取标记数据
    yuche =''
    bianhao = iamge_path.split('__', )[2]
    im = Image.open(iamge_path)  # 1.打开图片
    im = im.convert('L')  # 灰度处理
    binaryImage = im.point(initTable(), '1')  # 二值化
    binaryImage = noise_reduction(binaryImage)  # 去噪点
    crop_list = smartSliceImg(binaryImage)  # 返回切割坐标
    for box in crop_list:
        binaryImage.crop(box).save('./Test_temp/temp.png')
        data = getletter('./Test_temp/temp.png')
        data = np.array([data])
        oneLetter = clf.predict(data)[0]
        yuche += str(oneLetter) + '_'


    if yuche[:-1] ==each_name:
        print('预测值：',yuche[:-1],'实际值：',each_name,'-------------识别成功')
        return True
    else:
        print('预测值：',yuche[:-1],'实际值：',each_name,'-------------识别失败')
        return False

def TestALLSVM(path):
    file_name_list = get_file_name(path)
    total =len(file_name_list)
    total_success =0

    for  i in file_name_list:

        # print(path+'/'+i)
        success = TestSVM(path + '/' + i)
        if success:
            total_success+=1
    print('识别率',float(total_success/total))




        # Train_Pretreatment(path+'/'+i)





    # data = getletter(os.path.join('./Cutting/0','0_5.png'))




if __name__ == '__main__':
    chinese = {
        '丘': 0,
        '匆': 1,
        '白': 2,
        '他': 3,
        '甩': 4,
        '丛': 5,
        '仗': 6,
        '斥': 7,
        '禾': 8,
        '付': 9,
        '令': 10,
        '乐': 11,
        '乎': 12,
        '用': 13,
        '四': 14,
        '句': 15,
        '仪': 16,
        '瓜': 17,
        '册': 18,
        '生': 19,
        '仙': 20,
        '失': 21,
        '印': 22,
        '仔': 23,
        '们': 24,
        '代': 25,
    }
    # path = "./train_data/2019y07m05d_10h13m20s__0_0_1_2__0.png"
    # im = Image.open(path)  # 1.打开图片
    # im = im.convert('L')
    # binaryImage = im.point(initTable(), '1')
    # binaryImage = noise_reduction(binaryImage)
    # print_bin(binaryImage)  # 输出二值图像
    # smartSliceImg(binaryImage)

    #预处理+切割
    # path = './train_data/'
    # SliceImg(path)

    #训练

    # X,Y =load_dataset()
    #
    # trainSVM(X,Y)
#测试
    path ='./test_data/'
    TestALLSVM(path)
