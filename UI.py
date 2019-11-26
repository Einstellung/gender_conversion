import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

import torch
from models.CycleGAN_models import Generator
from torch.autograd import Variable
import torchvision.transforms as transforms

from models.wide_resnet import WideResNet
import numpy as np
import cv2
import dlib


class PneumoniaDetection(QWidget):
    def __init__(self, parent=None):
        super(PneumoniaDetection, self).__init__(parent)
        layout = QVBoxLayout()

        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)

        self.btn = QPushButton("导入图片")
        self.btn.clicked.connect(self.getfile)
        self.btn.setFont(font)
        layout.addWidget(self.btn)
        self.le = QLabel("")
        layout.addWidget(self.le)

        self.btn1 = QPushButton("男女性别转换")
        self.btn1.clicked.connect(self.detection)
        self.btn1.setFont(font)
        layout.addWidget(self.btn1)
        layout.addWidget(self.le)

        self.btn2 = QPushButton("导出生成好的图片")
        self.btn2.clicked.connect(self.output_picture)
        self.btn2.setFont(font)
        layout.addWidget(self.btn2)
        layout.addWidget(self.le)


        self.contents = QTextEdit()
        layout.addWidget(self.contents)
        self.setLayout(layout)
        self.contents.insertPlainText("注意：\n图片生成过程可能会有所卡顿")
        self.setWindowTitle("CycleGAN性别转换")

        self.fname = ""  # 用来保存文件名称，做预测的时候要用到这个文件名
        self.out_img = None

    def getfile(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, "打开图片", ".", "图像文件(*.jpg *.png *.jpeg)")


        

    def detection(self):
        depth = 16
        width = 8
        img_size = 64
        model = WideResNet(img_size, depth=depth, k=width)()
        model.load_weights(r'models/weights.hdf5')


        detector = dlib.get_frontal_face_detector()


        image_np = cv2.imdecode(np.fromfile(self.fname, dtype=np.uint8), -1)

        img_h = image_np.shape[0]
        img_w = image_np.shape[1]

        detected = detector(image_np, 1)

        gender_faces = []
        labels = []
        original_faces = []
        photo_position = []

        change_male_to_female_path = r'models/netG_A2B.pth'
        change_female_to_male_path = r'models/netG_B2A.pth'

        # 加载CycleGAN模型
        netG_male2female = Generator(3, 3)
        netG_female2male = Generator(3, 3)


        netG_male2female.load_state_dict(torch.load(change_male_to_female_path, map_location='cpu'))
        netG_female2male.load_state_dict(torch.load(change_female_to_male_path, map_location='cpu'))

        # 设置模型为预测模式
        netG_male2female.eval()
        netG_female2male.eval()

        transform_list = [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        transform = transforms.Compose(transform_list)

        """
        这段内容为图片数据处理frame
        """
        if len(detected) > 0:
            for i, d in enumerate(detected):
                # weight和height表示原始图片的宽度和高度，因为我们需要不停地resize处理图片
                # 最后要贴回到原始图片中去，w和h就用来做最后的resize
                x0, y0, x1, y1, w, h = d.left(), d.top(), d.right(), d.bottom(), d.width(), d.height()


                x0 = max(int(x0 - 0.25 * w), 0)
                y0 = max(int(y0 - 0.45 * h), 0)
                x1 = min(int(x1 + 0.25 * w), img_w - 1)
                y1 = min(int(y1 + 0.05 * h), img_h - 1)
                w = x1 - x0
                h = y1 - y0
                if w > h:
                    x0 = x0 + w // 2 - h // 2
                    w = h
                    x1 = x0 + w
                else:
                    y0 = y0 + h // 2 - w // 2
                    h = w
                    y1 = y0 + h
                    
                original_faces.append(cv2.resize(image_np[y0: y1, x0: x1, :], (256, 256)))
                gender_faces.append(cv2.resize(image_np[y0: y1, x0: x1, :], (img_size, img_size)))
                photo_position.append([y0, y1, x0, x1, w, h])  
                

            gender_faces = np.array(gender_faces)
            results = model.predict(gender_faces)
            predicted_genders = results[0]
            
            
            for i in range(len(original_faces)):
                labels.append('F' if predicted_genders[i][0] > 0.5 else 'M')
                
            for i, gender in enumerate(labels):
                
                # 这几个变量用于接下来图片缩放和替换
                y0, y1, x0, x1, w, h = photo_position[i]

                # 将数据转换成Pytorch可以处理的格式
                picture = transform(original_faces[i])
                picture = Variable(picture)
                input_picture = picture.unsqueeze(0)

                if gender == "M":
                    fake_female = 0.5*(netG_male2female(input_picture).data + 1.0)
                    out_img = fake_female.detach().squeeze(0)
                else:
                    fake_male = 0.5*(netG_female2male(input_picture).data + 1.0)
                    out_img = fake_male.detach().squeeze(0)
                    
                
                # 需要将Pytorch处理之后得到的数据，转换为OpenCV可以处理的格式
                # 下面代码就是转换代码
                image_numpy = out_img.float().numpy()
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                image_numpy = image_numpy.clip(0, 255)
                image_numpy = image_numpy.astype(np.uint8)       

                
                # 将转换好的性别图片替换到原始图片中去
                # 使用泊松融合使生成图像和背景图像浑然一体
                # 使用方法：cv2.seamlessClone(src, dst, mask, center, flags)
                generate_face = cv2.resize(image_numpy, (w, h))

                # Create an all white mask， 感兴趣的需要替换的目标区域，精确地mask可以更好的替换，这里mask就是生成图片的大小
                mask = 255 * np.ones((w, h), image_np.dtype)
                # center是目标影像的中心在背景图像上的坐标！
                center_y = y0 + h//2
                center_x = x0 + w//2
                center = (center_x, center_y)
                output_face = cv2.seamlessClone(generate_face, image_np, mask, center, cv2.NORMAL_CLONE)

                self.out_img = output_face


    def output_picture(self):
        self.fname, _ = QFileDialog.getSaveFileName(self, "保存图片", ".", "图像文件(*.jpg)")
        cv2.imencode(".jpg", self.out_img)[1].tofile(self.fname)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    analysis = PneumoniaDetection()
    analysis.show()
    sys.exit(app.exec_())
