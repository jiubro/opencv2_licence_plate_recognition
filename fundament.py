import cv2
import matplotlib.pyplot as plt
import numpy as np

img_address = "D:/python/pycharm/th.jpg"
vidio_address = 'D:/python/pycharm/wx.mp4'
# 显示色彩图片
def show1(address):
    img = cv2.imread(address)
    print(img)
    cv2.imshow('cat',img)
    cv2.waitKey(0)     # 卡死程序，参数设置为0即按任意键继续，设置为>1就记秒,毫秒为单位
    cv2.destroyAllWindows()
# 显示黑白图 && 利用方法显示尺寸
def show2(address):
    img = cv2.imread(address,cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    print(img)
    cv2.imshow('cat', img)  # 必须用wait才能一直显示
    cv2.waitKey(0)

# 保留图片
def save_(address):
    img = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('mycat.png',img)

# 显示视频
def show_vidio(address):
    vc = cv2.VideoCapture(address)

    open, frame = vc.read()
    while open != False:
        open,frame = vc.read()
        if open == True:
            vidio = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2.imshow('result',vidio)
            if cv2.waitKey(15) & 0xFF == 65:
             # waitkey函数有返回值，返回获取到键盘信号的ascii码值。而0xFF（其实就是二进制的11111111）的作用就是获取返回值的最后8位
             # 然后判断是否为ascii码值，这里的27对应 esc键的码值
                break
    vc.release()    # 需要释放内存
    cv2.destroyAllWindows()

# 图片切割（感兴趣区）
# 参数分别是行数和列数，所包围区域是切割后区域
def ROL_(row1,row2,col1,col2):
    vc = cv2.imread(img_address)
    cutted_ROl = vc[row1:row2,col1:col2,1]
    cv2.imshow('cat',cutted_ROl)
    cv2.waitKey(0)

# 颜色通道提取

def abstract():
    img = cv2.imread(img_address)
    b,g,r = cv2.split(img)  # 可以对图片列表进行切片
    temp = cv2.merge((b,g,r))
    print("切割后",g)
    print('合并后', temp)
    print(img)
    # 提取红色
    img[:,:,0] = 0
    img[:,:,1] = 0
    cv2.imshow('R channel',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 边界填充
def border_filling():
    img = cv2.imread(img_address)
    top_,bottom_,left_,right_ = (50,50,50,50)
    # 其实就是一个函数，最后面选择的模式不一样
    # BORDER_REPLICATE:复制法，也就是复制最边缘像素。
    # ·BORDER_REFLECT:反射法，对感兴趣的图像中的像素在两边进行复制 fedcba | abcdefgh | hgfedcb
    # BORDER_REFLECT101，上一种方法的优化版，去掉了边界 cdefgh|abcdefgh|abcdefg
    # BORDER_WRAP: 外包装法cdefgh|abcdefgh|abcdefg
    # BORDER_CONSTANT:常量法，常数值填充
    replicate = cv2.copyMakeBorder(img,top_,bottom_,left_,right_,cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img,top_,bottom_,left_,right_,cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img,top_,bottom_,left_,right_,cv2.BORDER_REFLECT101)
    wrap = cv2.copyMakeBorder(img,top_,bottom_,left_,right_,cv2.BORDER_WRAP)
    const = cv2.copyMakeBorder(img,top_,bottom_,left_,right_,cv2.BORDER_CONSTANT,value=0)
    cv2.imshow('1',replicate)

    cv2.imshow('2',reflect)

    cv2.imshow('3',reflect101)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    def image_mix():
        img1 = cv2.imread(img_address)
        print(img1.shape)
        img2 = cv2.imread("D:/python/pycharm/dog.jpg")
        img2 = cv2.resize(img2, (244, 240))  # 重新设置图片尺寸，要图片融合必须尺寸一样，注意这里的参数是宽和高——行列
        # print(img2.shape)
        #  img3 = img1 + img2  可以加，也就是遵循矩阵加法，但是不要简单相加
        #  print(img3)
        #  cv2.imshow('mix',img3)
        # cv2.waitKey(0)
        res = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)  # 图片占的权重
        cv2.imshow('mix', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# show2(img_address)
# show_vidio()
# ROL_(120,240,0,240)
# abstract()
# border_filling()

# 腐蚀操作
class corrosion:
    # 正常腐蚀操作，突发奇想二值反转
    def corrosion_1(self):
        img0 = cv2.imread("D:/python/pycharm/corrosion.jpg")
        img = cv2.resize(img0, (300, 300))  # 重新设定大小
        # 二值反转
        # 由于腐蚀操作目前只能腐蚀黑底白字，所以想自己写一个二值反转，显然图像是一个三维列表，可遍历
        # 法1 遍历法 一般会卡比较久，因为复杂度高，哈哈  )+_+(
        # for i in range(len(img)):
        #     for j in range(len(img[i])):
        #         for k in range(len(img[i][j])):
        #             if img[i][j][k] <= 256 and img[i][j][k] >= 200:
        #                 img[i][j][k] = 0
        #             elif img[i][j][k] < 200 and img[i][j][k] >= 0:
        #                 img[i][j][k] = 255

        # 法2 库函数
        img = cv2.bitwise_not(img)
        print(img)
        print(img.shape)
        kernel = np.ones((5, 5), np.uint8)  # 画卷积盒，用于限定腐蚀大小,当然也可以画不止矩形，这个可以自己发掘
        img = cv2.erode(img, kernel, iterations=1)  # 主要腐蚀函数 图像，卷积盒 ，迭代次数
        cv2.imwrite("D:/python/pycharm/binary-literal.jpg",img)
        cv2.imshow('erosion', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # 正常图片转灰度
    def corrsion_2(self):
        img = cv2.imread(img_address,cv2.IMREAD_GRAYSCALE)  # 最关键的是这个函数的参数
        cv2.imshow('gray',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # 图片转二值
    def corrsion_3(self):

        img = cv2.imread(img_address, cv2.IMREAD_GRAYSCALE)   # 先转灰度
        # 图像二值化
        # 127设置的是阈值，超过该阈值则设定为255——白色，小于设置为0——黑色
        # 返回值，第一个返回值是二值化最佳阈值，第二个是像素列表
        thresh_value,img2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        thresh_value2, img2 = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)  # 使用最佳阈值
        print(thresh_value)     # 其实就是127啦 ==__==
        # 保存二值化后的图片
        cv2.imwrite("D:/python/pycharm/binary-cat.png", img2)


# 膨胀操作,与腐蚀操作相反，这里不赘述
def dilate_():
    im1 = cv2.imread("D:/python/pycharm/binary.png")
    im2 = cv2.imread("D:/python/pycharm/dog.jpg")
    im3 = cv2.imread("D:/python/pycharm/th.jpg")

    kernel = np.ones((5, 5), np.uint8)

    # 膨胀操作后图片拼接
    # 注意图片大小相同，resize修改大小
    im1 = cv2.resize(im1,(300,300))
    im2 = cv2.resize(im2,(300,300))
    im3 = cv2.resize(im3,(300,300))

    im1 = cv2.dilate(im1,kernel,iterations=1)
    im2 = cv2.dilate(im2,kernel,iterations=1)
    im3 = cv2.dilate(im3,kernel,iterations=1)
    # 拼接函数
    res  = np.hstack((im1,im2,im3))
    cv2.imshow('res',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 开运算与闭运算、梯度、礼帽与黑帽
# 开：先腐蚀再膨胀，————弥补了消去刺或杂点带来的对原字体的影响，闭：先膨胀再腐蚀————加大刺
# 梯度是开运算后图像减闭运算后图像，得到边界轮廓，像镂空字体
# 礼帽 = 原始输入 - 开运算结果
# 黑帽 = 闭运算 - 原始输入
def morphology_():
    im0 = cv2.imread("D:/python/pycharm/chepai_test.png",cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3,3),np.uint8)
    im1 = cv2.morphologyEx(im0,cv2.MORPH_OPEN,kernel)   # 开运算
    im2 = cv2.morphologyEx(im0,cv2.MORPH_CLOSE,kernel)  # 闭运算
    im3 = cv2.morphologyEx(im0,cv2.MORPH_GRADIENT,kernel)   # 梯度计算
    im4 = cv2.morphologyEx(im0,cv2.MORPH_TOPHAT,kernel)
    im5 = cv2.morphologyEx(im0,cv2.MORPH_BLACKHAT,kernel)
    res = np.hstack((im0,im1,im2,im3,im4))
    cv2.imshow('opening-closing',res)
    cv2.imshow('blackhat',im5)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图片显示
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 多张图片拼接显示
# 注意拼接的时候位数要一样，比如灰度图是一维不可以和彩色图拼接，可以选择升维，链接https://blog.csdn.net/qq_43391414/article/details/111085167
def cv_show_numbers(name,im0,im1,im2=[],im3=[],im4=[]):
    if len(im2)==0:
        res = np.hstack((im0, im1))
    elif len(im2) !=0 and len(im3)==0 :
        res = np.hstack((im0, im1,im2))
    elif len(im3) !=0 and len(im4)==0 :
        res = np.hstack((im0,im1,im2,im3))
    elif len(im3) !=0:
        res = np.hstack((im0, im1,im2,im3,im4))    # 也可以vstack是垂直显示

    cv2.imshow(name, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# sobel算子----其实就是边缘检测常用，利用梯度差异，当一个像素点某个方向有梯度差异，利用sobel算子的处理方法将会重新赋值这个点的像素值
#dst = cv2.Sobel(src, ddepth, dx, dy, ksize)
#. ddepth:图像的深度,一般输入-1，-1代表输入深度与输出一致
#·dx和dy分别表示水平和竖直方向. 设置为0就是不计该方向ksize是Sobel算子的大小

def sobel():
    im = cv2.imread("D:/python/pycharm/filter.jpg",cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    # 将x方向和y方向的算子用函数求权重，不建议直接设置Sobel函数xy方向都是1，那样效果不好
    sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)  # xy的权重，0是表示权重，这里设0
    cv_show(sobelxy,'soblexy')

# 介绍另外两种算子，scharr算子更能描绘细节，而laplacian算子对噪音点敏感，但不一定对边界敏感
def Scharr_laplacian():
    im = cv2.imread(img_address,cv2.IMREAD_GRAYSCALE)
    # scharrx算子
    scharrx = cv2.Scharr(im,cv2.CV_64F,1,0)
    scharry = cv2.Scharr(im,cv2.CV_64F,0,1)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.convertScaleAbs(scharry)
    scharrxy = cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
    # 拉普拉斯算子,不需要取xy两个方向
    laplacian = cv2.Laplacian(im,cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    cv_show_numbers('Scharr---laplacian',scharrxy,laplacian)

# 滤波
def Filter_():
    # 均值滤波,简而言之就是取周围像素平均值。围绕目标点取一定大小卷积盒，该卷积盒与对应像素点乘积的均值为该点像素，以此为该点像素值
    # 是最基础的一种滤波
    im = cv2.imread("D:/python/pycharm/filter.jpg")
    avg = cv2.blur(im,(3,3))
    # 方框滤波，求周围像素之和，但是不求均值，当参数设置为归一化时效果与均值一致
    # 有一个参数可以设置归一化，为True则归一化————大于255截断重来，这样的话效果和均值一样，为false大于255置255
    box1 = cv2.boxFilter(im,-1,(3,3),normalize=True)
    box2 = cv2.boxFilter(im,-1,(3,3),normalize=False)
    cv_show_numbers('filter',im,avg,box1,box2)

# 高斯滤波————离得越近的像素点占比越大
def Gaoshi_median_Filter():
    im = cv2.imread("D:/python/pycharm/filter.jpg")
    # 高斯
    aussian = cv2.GaussianBlur(im,(5,5),1)
    # 中值
    median = cv2.medianBlur(im,5)
    cv_show_numbers('img',im,aussian,median)

# 边缘检测——Canny算法
def Canny_():
    im = cv2.imread("D:/python/pycharm/chepai_test.png",cv2.IMREAD_GRAYSCALE)
    im1 = cv2.Canny(im,50,100)
    im2 = cv2.Canny(im,80,150)
    im3 = cv2.Canny(im,150,250)
    cv_show_numbers("Canny_",im1,im2,im3)

# 轮廓检测及输出。前提是先把图像转化为二值图像，其实很多图像特征的提取都要化成二值，比如边缘检测
def contours_():
    im = cv2.imread("D:/python/pycharm/meinv.jpg")
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    # 返回两个值，第一个值是返回轮廓信息，用作画框函数的参数
    # 传入参数第一个是二值图像，第二个是检测轮廓模式，第三个是画框模式
    contours,hierarachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # 这里为什么要复制呢，主要是下面drawcontours函数会处理传入的图片，让它直接为操作结果，覆盖这个图像
    draw_im = gray.copy()
    # im——传入要画轮廓的图像，这里传入的是初始彩色图像，在彩色图像上绘制
    # contours——传入上面find函数的返回值，也就是轮廓信息，
    # -1表示画所有框，（0,0,255）是框的颜色，2是线条厚度
    res = cv2.drawContours(im,contours,-1,(0,0,255),2)

    # 这里不要用我们自定义的拼接显示，因为彩色图和灰度图维数不一样
    cv_show(res,'dasb')

# 轮廓特征与轮廓近似
def contours_2():
    im = cv2.imread("D:/python/pycharm/pentagram.jpg")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imshow("init",thresh)
    cnt = contours[0]      # 传入单个轮廓
    temp = im.copy()
    res = cv2.drawContours(temp, cnt, -1, (0, 0, 255), 2)

    print(list(zip([1,],cnt))[:2])
    # 打印轮廓面积
    print(cv2.contourArea(cnt))
    # 打印轮廓周长
    print(cv2.arcLength(cnt,True))
    print(ret)

    # 轮廓近似
    # 1）、设置精度（从轮廓到近似轮廓的最大距离）
    epsilon =   0.09*cv2.arcLength(cnt, True)   # 这个参数要调，调节到近似框住
    #                             轮廓  闭合轮廓还是曲线
    # 2）、获取近似轮廓
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    #                             近似度(这里为10%)   闭合轮廓还是曲线
    print(approx)
    temp2 = im.copy()
    res2 = cv2.drawContours(temp2, [approx], -1, (0, 0, 255), 2)
    # 边界矩形--其实就是用获得边界信息，然后再用画矩形函数画一个矩形
    x,y,w,h = cv2.boundingRect(cnt)
    # 这个函数依然会改变原图像，所以继续拷贝
    temp3 = im.copy()
    img = cv2.rectangle(temp3,(x,y),(x+w,y+h),(0,0,255),2)  # 这里的第二第三个参数传入的左上右下坐标

    cv_show(res2,'approxpoly')
    cv_show(temp3,'img')    # 这里我们可以直接show传入的temp3，毕竟原图像本就被改变了

# 模板匹配
def template_match():
    img0 = cv2.imread("D:/python/pycharm/youhua.jpg")
    img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    template0 = cv2.imread("D:/python/pycharm/youhuaface.jpg")
    template = cv2.cvtColor(template0, cv2.COLOR_BGR2GRAY)
    h,w = img.shape[:2]
    methods = ['cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    print('res',res.sum)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    (startX, startY) = min_loc
    endX = startX + template.shape[1]
    endY = startY + template.shape[0]
    #在图像上绘制边框
    cv2.rectangle(img0, (startX, startY), (endX, endY), (255, 0, 0), 3)
    #显示输出图像
    cv2.imshow("Output", img0)
    cv2.waitKey(0)
    cv2.imshow("Output", res)
    cv2.waitKey(0)
# 图像金字塔,了解其原理就知道这里不需要二值化或灰度化
def pyramid_():
    im = cv2.imread("D:/python/pycharm/youhua.jpg")
    # 高斯金字塔
    up = cv2.pyrUp(im)

    #  拉普拉斯
    # 迭代缩小放大
    down = cv2.pyrDown(im)
    im2 = cv2.pyrUp(down)
    # 不清楚为什么会多一行
    im2 = cv2.resize(im2,(241,306))
    lapla = im - im2
    print(im2.shape)
    print(im.shape)


    cv2.imshow('init',im)
    cv2.imshow('up',up)
    cv2.imshow('dowm',down)
    cv2.imshow('lapla',lapla)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# corrosion1. corrosion_1()
# dilate_()
# border_filling()
# morphology_()
# sobel()
# Scharr_laplacian()
# Filter_()
# Gaoshi_median_Filter()
#Canny_()
# contours_2()
# template_match()
# pyramid_()
# print(type())
