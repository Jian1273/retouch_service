class Config:
    def __init__(self) -> None:
        self.retouch_degree = 3  # 磨皮程度
        self.details_degree = 1  # 细节程度
        self.dx = self.retouch_degree*5  # 双边滤波参数之一
        self.fc = self.retouch_degree*12.5  # 双边滤波参数之二
        self.p = 0.1  # 双边滤波权值
        self.sharpness = 1.0  # 锐度
        self.contrast = 1.0  # 对比度
