class Container:
    isinstance: "Container" = None
    is_exist: bool = False

    def __new__(cls, *args, **kwargs):
        if not cls.is_exist:
            cls.instance = super().__new__(cls, *args, **kwargs)
            cls.is_exist = True
        return cls.instance

    def __init__(self):
        if self.is_exist: return