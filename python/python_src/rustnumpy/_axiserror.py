class AxisError(ValueError, IndexError):
    def __init__(self, axis, ndim=None, msg_prefix=None):
        if ndim is None and msg_prefix is None:
            msg = axis
        else:
            msg = "axis %s is out of bounds for array of dimension %s" % (axis, ndim)
            if msg_prefix is not None:
                msg = "%s: %s" % (msg_prefix, msg)
        super().__init__(msg)
        self.axis = axis
        self.ndim = ndim
