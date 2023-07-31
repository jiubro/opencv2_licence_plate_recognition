import cv2
import numpy as np
# 图片显示
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cutting_(thresh,contours):
    name = 0
    img_all = []
    station = []
    for c in contours:
        # 边界矩形--其实就是用获得边界信息，然后再用画矩形函数画一个矩形
        x, y, w, h = cv2.boundingRect(c)
        station.append(x)
        img = thresh[y:y+h,x:x+w]
        img_all.append(img)
    img_all = [img_temp  for _, img_temp in sorted(zip(station,img_all))]

    for t in img_all:
        cv2.imwrite(f"D:/python/{name}.jpg",t)
        name += 1
    return img_all

def being_binary(address,thresh,maxval,model):
    img = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
    best_thresh, thresh = cv2.threshold(img, thresh, maxval, model)
    return img,thresh



img = cv2.imread("D:/python/pycharm/chepai_model.png")
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 转化为黑底白字，因为膨胀腐蚀处理只能黑底白字
best_thresh,thresh = cv2.threshold(ref,127, 255, cv2.THRESH_BINARY)
# 寻找轮廓函数
contours, hierarachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 显示轮廓
temp = img.copy()
cv2.drawContours(temp, contours, -1, (0, 0, 255), 2)

# 轮廓个数,如果轮廓个数不对一定要重视，看一看是不是图片的细节没有处理好，边缘的颜色和目标颜色相同
# print(np.array(contours).shape,best_thresh,img.shape,ref.shape)
img_all = cutting_(thresh,contours)

# 输入图像地址
input_address = "D:/python/pycharm/chepai_test2.png"
# 输入图像处理
input_img,thresh2 = being_binary(input_address,127, 255, cv2.THRESH_BINARY)
null_img,thresh_null = being_binary("D:/python/pycharm/chepai_null.png",127, 255, cv2.THRESH_BINARY)
y,x = null_img.shape
kernel = np.ones((5, 5), np.uint8)
null_img = cv2.dilate(null_img,kernel,iterations=4)
thresh2 = cv2.resize(thresh2,(x,y))
thresh2 = cv2.subtract(thresh2, null_img)
# 寻找轮廓函数,通过：空外框膨胀与原图像做差；面积排序找出最大的7个图像面积，也即对应的以此获取输入图像的轮廓

contours2, hierarachy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
input0 = cv2.imread(input_address)

area_all = []
for cnt in contours2:
    area = cv2.arcLength(cnt, True)
    area_all.append(area)
# zip完了之后如果是两个不同的数据类型记得强制指定一下数据类型，要不然不男不女
zip_contours2 = sorted(zip(area_all,list(range(len(area_all)+1))),reverse=True)[:8]
print(zip_contours2)
for _,i in zip_contours2:
    cv2.drawContours(input0, contours2[i], -1, (0, 0, 255), 2)

# 因为车牌全为数字的话一般只有7位，取面积最大的7个就可以了

cv_show("s",thresh2)
print(type(input_img),type(img_all))


# 再次读取输入图像
input_draw = cv2.imread(input_address)
# 输入图像裁剪成7个并与10个模板匹配
for contour in contours2:
    x, y, w, h = cv2.boundingRect(contour)  # 获取轮廓的外接矩形
    digit_roi = thresh2[y:y + h, x:x + w]  # 从图像中提取数字的感兴趣区域
    digit_roi = cv2.copyMakeBorder(digit_roi, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=0)
    min_score = 100
    min_match_index = None
    for i in range(len(img_all)):

        result_ = cv2.matchTemplate(digit_roi, img_all[i], cv2.TM_SQDIFF_NORMED)
        match_score, _, _, _ = cv2.minMaxLoc(result_)

        if match_score < min_score:
            min_score = match_score
            min_match_index = i

    cv2.putText(input_draw, f'{min_match_index}',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



cv_show('THT',input_draw)



