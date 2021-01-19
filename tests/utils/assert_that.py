class AssertThat():
    def __init__(self, x=None,y=None):
        self.x = x
        self.y = y
    
    def are_equal(self):
        return self.x == self.y

    def are_not_equal(self):
        return self.x != self.y

    def is_in_interval(self, a, b):
        return a < self.x < b