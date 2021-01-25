class Function:  # A differentiable function

    def __init__(self, *args, **kwargs):
        # cache to use in backward prop (input and output of forward pass)
        self.cache = {}
        # cache to use in backward prop (gradient of forward pass)
        self.grad = {}

    def __call__(self, *args, **kwargs):
        # output of this function forward pass
        self.out = self.forward_pass(*args, **kwargs)
        # caching output of this function local gradient
        self.grad = self.local_gradient(*args, **kwargs)
        '''
         Here I cached the grad and will cache the input and
         output of the forward pass later in forward_pass function
        '''
        return self.out

    def forward_pass(self, *args, **kwargs):
        '''
        Later implemented for each function specifically.
        Computes forward pass of the function and cache it.
        '''
        pass

    def local_gradient(self, *args, **kwargs):
        '''
        Later implemented for each function specifically.
        Computes local gradient of the function and return dictionary of them.
        '''
        pass

    def backward_pass(self, *args, **kwargs):
        '''
        Later implemented for each function specifically.
        Computes global gradient of the function.
        '''
        pass