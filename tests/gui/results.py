class Results:
    def __init__(self):
        self.cnn = []
        self.pca = []
        self.vgg = []
        self.sift = []

    def set_cnn(self, cnn):
        self.cnn = cnn

    def get_cnn(self):
        return self.cnn

    def set_pca(self, pca):
        self.pca = pca

    def get_pca(self):
        return self.pca

    def set_vgg(self, vgg):
        self.vgg = vgg

    def get_vgg(self):
        return self.vgg

    def set_sift(self, sift):
        self.sift = sift

    def get_sift(self):
        return self.sift
