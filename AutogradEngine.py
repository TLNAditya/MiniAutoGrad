import math
class Value:
    def __init__(self,data,_children=(),_op=""): 
        self.data = data
        self.grad = 0.0
        self._prev = set(_children) 
        self._backward = lambda:None
        self._op = _op

    def __repr__(self):
        return f'Value(data= {self.data})'

    def __add__(self,other):
        out = Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out 

    def __mul__(self,other):
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad +=(1-t**2)*out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0,self.data),'ReLU')
        def _backward():
            if(out.data==self.data):
                self.grad += 1
            else:
                self.grad += 0
        out._backward = _backward
    
    def sigmoid(self):
        out = Value(1/(1+math.exp(-self.data)),(self,),'Sigmoid')
        def _backward():
            self.grad += out.data*(1-out.data)
        out._backward = _backward

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)#here o->self
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()