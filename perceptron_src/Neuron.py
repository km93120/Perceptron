import random

class Neuron(object):

    def __init__(self, position, value=0, active=False):
        # self.value = random.randint(-1, 1)
        self.value = value
        self.xPos = position[0]
        self.yPos = position[1]
        self.active = active

    def __str__(self):
        return "({},{}) : {} | active={}".format(self.xPos, self.yPos, 
                                                 self.value, self.active)
    
    __repr__ = __str__

    def __eq__(self, other):
        return (self.value == other.value and 
                self.xPos == other.xPos and
                self.yPos == other.yPos and
                self.active == other.active)

    def __lt__(self, other):
        if self.yPos == other.yPos:
            return self.xPos < other.xPos
        else:
            return self.yPos < other.yPos
