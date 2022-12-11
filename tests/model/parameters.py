class Parameters:
    def __init__(self):
        self.cnn = []
        self.pca = []
        self.vgg = []
        self.sift = []

    def set_cnn(self, cnn):
        self.cnn = cnn

    def get_cnn(self):
        return self.cnn

    def reset_cnn(self):
        self.cnn.clear()

    def set_pca(self, pca):
        self.pca = pca

    def get_pca(self):
        return self.pca

    def reset_pca(self):
        self.pca.clear()

    def set_vgg(self, vgg):
        self.vgg = vgg

    def get_vgg(self):
        return self.vgg

    def reset_vgg(self):
        self.vgg.clear()

    def set_sift(self, sift):
        self.sift = sift

    def get_sift(self):
        return self.sift

    def reset_sift(self):
        self.sift.clear()
