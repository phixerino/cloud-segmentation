
class AttrDict(dict):
    """Class that acts like a dictionary + items can be accessed by attribute"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

