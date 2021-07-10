import torch
import torch.nn as nn

class MyRecurrentNet(nn.Module):
    def __init__(self, input_dim, batch_size=1):
        super().__init__()
        self.out_size = 10
        self.hidden = 5
        self.U = nn.Parameter(torch.ones(self.hidden, input_dim))
        self.bias = nn.Parameter(torch.zeros(self.hidden, input_dim))

        self.h0 = nn.Parameter(torch.zeros(input_dim, self.hidden))
        self.W = nn.Parameter(torch.ones(self.hidden, self.hidden))

        self.tanh = nn.Tanh()
        #self.batch_size = 
        self.V = nn.Parameter(torch.ones(self.out_size, self.hidden))
        

    def forward(self, x):
        
        self.batch_size = x.shape[0]
        x_times_u = torch.mm(x, self.U.t())
        h_times_w = torch.mm(self.h0, self.W)
        z = h_times_w + x_times_u + self.bias.t()
        h = self.tanh(z)
        assert h.shape == self.h0.shape, str(self.h0.shape) + str(h.shape)
        o = torch.mm(h, self.V.t() )#
        
        #y_hat = torch.softmax(o)
        return o

class MyGatedRecurrentUnit (nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.hidden = 5
        self.h0 = nn.Parameter(torch.ones(input_dim, self.hidden))
        self.c0 = nn.Parameter(torch.ones(input_dim, self.hidden))
        #self.h0 = nn.Parameter(torch.ones(self.hidden, self.hidden))
        # z gate
        self.Wz = nn.Parameter(torch.ones(self.hidden, input_dim))
        self.Uz = nn.Parameter(torch.ones(self.hidden, self.hidden))
        self.Vz = nn.Parameter(torch.ones(input_dim, self.hidden))
        self.biasz = nn.Parameter(torch.ones(self.hidden))
        # r 
        self.Wr = nn.Parameter(torch.ones(self.hidden, input_dim))
        self.Ur = nn.Parameter(torch.ones(self.hidden, self.hidden))
        self.Vr = nn.Parameter(torch.ones(input_dim, self.hidden))
        self.biasr = nn.Parameter(torch.ones(self.hidden))
        # h_hat
        self.Wh = nn.Parameter(torch.ones(self.hidden, input_dim))
        self.Uh = nn.Parameter(torch.ones(self.hidden, self.hidden))
        self.Vh = nn.Parameter(torch.ones(input_dim, self.hidden))
        self.biash = nn.Parameter(torch.ones(self.hidden))

    def forward(self, x):
        # z
        z = self.sigmoid(torch.mm(x, self.Wz.t()) + 
                        torch.mm(self.h0, self.Uz.t()) + 
                        #self.Vz * self.c0 + 
                        self.biasz)
        
        # r
        r = self.sigmoid(torch.mm(x, self.Wr.t()) + 
                        torch.mm(self.h0, self.Ur.t()) + 
                        #self.Vr * self.c0 + 
                        self.biasr)

        # h_hat
        h_hat = self.tanh(torch.mm((r * self.h0), self.Uh.t()) + torch.mm(x, self.Wh.t()))
        # h 
        h = (1 - z) * self.h0 + z * h_hat
        #print(z.shape)
        return h


class MyLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.hidden = 5
        self.h0 = nn.Parameter(torch.ones(input_dim, self.hidden))
        self.c0 = nn.Parameter(torch.ones(input_dim, self.hidden))
        #self.h0 = nn.Parameter(torch.ones(self.hidden, self.hidden))
        # input gate
        self.Wi = nn.Parameter(torch.ones(self.hidden, input_dim))
        self.Ui = nn.Parameter(torch.ones(self.hidden, self.hidden))
        self.Vi = nn.Parameter(torch.ones(input_dim, self.hidden))
        self.biasi = nn.Parameter(torch.ones(self.hidden))
        # forget gate 
        self.Wf = nn.Parameter(torch.ones(self.hidden, input_dim))
        self.Uf = nn.Parameter(torch.ones(self.hidden, self.hidden))
        self.Vf = nn.Parameter(torch.ones(input_dim, self.hidden))
        self.biasf = nn.Parameter(torch.ones(self.hidden))
        # output gate
        self.Wo = nn.Parameter(torch.ones(self.hidden, input_dim))
        self.Uo = nn.Parameter(torch.ones(self.hidden, self.hidden))
        self.Vo= nn.Parameter(torch.ones(input_dim, self.hidden))
        self.biaso = nn.Parameter(torch.ones(self.hidden))
        # cell calculation
        # output gate
        self.Wc = nn.Parameter(torch.ones(self.hidden, input_dim))
        self.Uc = nn.Parameter(torch.ones(self.hidden, self.hidden))
        #self.Vc= nn.Parameter(torch.ones(input_dim, self.hidden))
        self.biasc = nn.Parameter(torch.ones(self.hidden))

    def forward(self, x):
        # input gate
        i = self.sigmoid(torch.mm(x, self.Wi.t()) + 
                        torch.mm(self.h0, self.Ui.t()) + 
                        self.Vi * self.c0 + 
                        self.biasi)
        # forget gate
        # input gate
        f = self.sigmoid(torch.mm(x, self.Wf.t()) + 
                        torch.mm(self.h0, self.Uf.t()) + 
                        self.Vf * self.c0 + 
                        self.biasf)
        # output gate
        o = self.sigmoid(torch.mm(x, self.Wo.t()) + 
                        torch.mm(self.h0, self.Uo.t()) + 
                        self.Vo * self.c0 + 
                        self.biaso)
        # cell state                        
        c = f * self.c0 + i * self.tanh(torch.mm(x, self.Wc.t()) + 
                        torch.mm(self.h0, self.Uc.t()) + 
                        self.biasc)
        # hidden state
        h = o * self.tanh(c)
        # shouls return h, c, o 
        return h

# 𝒛(𝑡)=𝑾𝒉(𝑡−1)+𝐔𝒙(𝑡)+𝒃
# 𝒉(𝑡)=tanh𝒛𝑡𝒐(𝑡)=𝐕𝒉(𝑡)ෝ𝒚𝑡=softmax𝒐
x = torch.tensor([[3,4], [5,3]]).float()
net = MyLSTM(2)
net(x)
#print(net(x))
#print(x.shape)
torch.ones(2) + torch.ones(2,2)


