from cv2 import addWeighted, waitKey
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
from config import Config

class skinRetouch:
    def __init__(self, config) -> None:
        self.config = config

    def blurred(self, path):
        '''
        模糊处理
        Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100
        '''
        img = cv2.imread(path)
        dst = np.zeros_like(img)
        temp4 = np.zeros_like(img)
        temp1 = cv2.bilateralFilter(img,self.config.dx,self.config.fc,self.config.fc)
        temp2 = cv2.subtract(temp1,img)
        temp2 = cv2.add(temp2,(10,10,10,128))
        temp3 = cv2.GaussianBlur(temp2,(2*self.config.details_degree - 1,2*self.config.details_degree-1),0)
        temp4 = cv2.add(img,temp3)
        dst = cv2.addWeighted(img,self.config.p,temp4,1-self.config.p,0.0)
        dst = cv2.add(dst,(10, 10, 10,255))
        save_path = path.replace(".jpg", "_blurred.jpg")
        cv2.imwrite(save_path, dst)
        return save_path
    
    def remove_spot(self, path):
        # 祛斑
        # sobel算子边缘检测 找到斑点可能存在的位置
        img = cv2.imread(path)

        temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转为灰度图
        x_gray = cv2.Sobel(temp, cv2.CV_32F, 1, 0)
        y_gray = cv2.Sobel(temp, cv2.CV_32F, 0, 1)
        x_gray = cv2.convertScaleAbs(x_gray)
        y_gray = cv2.convertScaleAbs(y_gray)
        dst = cv2.add(x_gray, y_gray, dtype=cv2.CV_16S)
        dst = cv2.convertScaleAbs(dst)

        # 连通域分析，确定斑点所在的位置
        num_labels,labels,stats,centers = cv2.connectedComponentsWithStats(dst, connectivity=8,ltype=cv2.CV_32S)
        # num_labels： 代表连通域的数量，包含背景
        # labels ： 记录img中每个位置对应的label
        # stats： 每个连通域的外接矩形和面积
        # x, y, w, h, area = stats[t]
        # centers : 连通域的质心坐标
        for t in range(1, num_labels, 1):
            x, y, w, h, area = stats[t]
            # print(x, y, w, h)
            if area>100:
                index = np.where(labels==t)
                labels[index[0], index[1]] = 0
        mask = np.array(labels, np.uint8)
    
        # 图像修复，去除斑点
        dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
        save_path = path.replace(".jpg", "_remove_spot.jpg")
        cv2.imwrite(save_path, dst)

    def retouch(self, path, id=0):
        """
        锐化、对比度调节
        """
        image = Image.open(path)

        if id == 0:
            # 锐度调节
            enh_img = ImageEnhance.Sharpness(image)
            image_sharped = enh_img.enhance(1.8)
            # 对比度调节
            con_img = ImageEnhance.Contrast(image_sharped)
            image_con = con_img.enhance(1.15)    
        elif id == 1:
            # 锐度调节
            enh_img = ImageEnhance.Sharpness(image)
            image_sharped = enh_img.enhance(1.8)
            # 对比度调节
            con_img = ImageEnhance.Contrast(image_sharped)
            image_con = con_img.enhance(1.10)
        else:
            pass
            
        save_path = path.replace(".jpg", "_retouch.jpg")
        image_sharped.save(save_path)
        return save_path

    def blurred_with_mask(self, orgin_path, mask_matrix, kernal,num):
        '''
            origin_path: 原图路径
            mask_matrix: 掩码矩阵
            kernal: 模糊矩阵
            num: 磨皮连接区域精细度
        '''
        img_0, img_1, img_mask = cv2.imread(orgin_path), cv2.imread(orgin_path), mask_matrix
        # fcn获取人脸具体区域，将非人脸区域设置为0
        img_1[mask_matrix != 13] = 0
        # 对mask区域的图像进行磨皮操作
        dst = np.zeros_like(img_1)
        temp4 = np.zeros_like(img_1)
        temp1 = cv2.bilateralFilter(img_1,self.config.dx,self.config.fc,self.config.fc)
        temp2 = cv2.subtract(temp1,img_1)
        temp2 = cv2.add(temp2,(10,10,10,128))
        temp3 = cv2.GaussianBlur(temp2,(2*self.config.details_degree - 1,2*self.config.details_degree-1),0)
        temp4 = cv2.add(img_1,temp3)
        dst = cv2.addWeighted(img_1,self.config.p,temp4,1-self.config.p,0.0)
        dst = cv2.add(dst,(10, 10, 10,255)) 
        
        for i in range(10):
            img_mask_ring = img_mask.copy()
            # 通过图像腐蚀得到环状区域
            img_mask_erode = cv2.erode(img_mask, kernel=kernal)
            img_mask_ring[img_mask_erode==13] = 0

            # 将环状区域的像素进行融合
            img_0[np.where(img_mask_ring==13)] = (1-i/num)*img_0[np.where(img_mask_ring==13)] + i/num*dst[np.where(img_mask_ring==13)]
            # mask区域更新
            img_mask = img_mask_erode
        
        img_0[img_mask==13] = 0
        # 磨皮过程中mask区域外的像素点还原
        dst[img_mask != 13] = 0 
        img_0 = cv2.add(img_0, dst)
        
        save_path = orgin_path.replace(".jpg", "_blur_mask.jpg")
        cv2.imwrite(save_path, img_0)
        # cv2.imwrite(save_path, img_0+dst)
        print("模糊完成",save_path)
        return save_path

# 连通域区分不同人的脸部区域
def distinguish_faces(mask_path):
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    print(mask.shape)
    # fcn获取人脸具体区域，将非人脸区域设置为0
    mask[mask != 13] = 0
    mask[mask == 13] = 255
    cv2.imwrite(mask_path.replace(".png", "_test.png"), mask)
    # 连通域分析
    num_labels,labels,stats,centers = cv2.connectedComponentsWithStats(mask, connectivity=8,ltype=cv2.CV_32S)
    # 判断连通域是否属于脸部，目前是通过连通域的面积来判断，大于10000认为是脸部的可能性较大
    faces = []
    print(num_labels)
    for t in range(1, num_labels, 1):
        img_mask = mask
        img_mask[labels== t] = 13
        img_mask[labels!= t] = 0
        if stats[t][4] >= 10000:
            faces.append(img_mask)
    print("不同人脸区分完成，一共有%s个人脸区域" % len(faces))
    return faces

def check_mask(mask_path):
     mask = cv2.imread(mask_path)
     mask[mask == 13] = 255
     cv2.imwrite(mask_path.replace(".png", "_check.png"),mask)


if __name__ == "__main__":
    config = Config()  # 导入配置参数
    sr = skinRetouch(config)
    # 两个脸部无连接
    orgin_path = "muti_faces/raw/XBP_6209.jpg"
    mask_path= "muti_faces/mask/XBP_6209.png"

    check_mask(mask_path)
    # 区分多人脸，返回多人的mask_matrix
    mask_matrixs = distinguish_faces(mask_path)
    # # 对每个人的mask_matrix进行磨皮
    # for mask_matrix in mask_matrixs:
    #     blur = sr.blurred_with_mask(orgin_path, mask_matrix, (3,3),12)
    #     retouch = sr.retouch(blur, 0)
        # print("磨皮完成")
        # sr.remove_spot(retouch)
        # print("去除斑点完成")

    # mask = np.array([[1,2,3,4],[13,13,12,11]])
    # mask[mask == 13] = 255
    # print(mask)