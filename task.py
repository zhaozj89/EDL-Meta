from sklearn.model_selection import train_test_split

class Task:
    def __init__(self, X, y, center_X, center_y, test_size=0.5):
        self.X = X
        self.y = y
        self.center_X = center_X
        self.center_y = center_y

        if X is not None:
            self.num, self.dim = X.shape
        else:
            _, self.dim = center_X.shape
            self.num = 0

        if X is None or len(X) == 1:
            self.X_train = X
            self.y_train = y
            self.X_test = X
            self.y_test = y
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size=test_size)

    def get_all(self):
        return self.X, self.y

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

    def get_num(self):
        return self.num

    def get_len(self):
        return self.dim

    def get_center(self):
        return self.center_X, self.center_y


class TestTask:
    def __init__(self, X, y, center_X, center_y):
        self.X = X
        self.y = y
        self.center_X = center_X
        self.center_y = center_y

        if X is not None:
            self.num, self.dim = X.shape
        else:
            _, self.dim = center_X.shape
            self.num = 0

    def get_all(self):
        return self.X, self.y

    def get_num(self):
        return self.num

    def get_len(self):
        return self.dim

    def get_center(self):
        return self.center_X, self.center_y
