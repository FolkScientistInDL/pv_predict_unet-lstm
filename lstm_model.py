import numpy as np
import random
import math
import torch.nn.functional as F
import os
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def attention_model(input_size, attention_size):
    model = nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size, 1, bias=False))
    return model

def attention_forward(atten_h_1,model_h0,model_hn,model, enc_states, dec_state):
    """
    enc_states: (time_step, bs, hidden_states)
    dec_state: (bs, hidden_states)
    """
    enc_states=atten_h_1(enc_states)
    if(enc_states.shape.__len__()==2):
        enc_states = enc_states.unsqueeze(dim=0)
    enc_and_dec_states = model_h0(enc_states)+model_hn(dec_state)
    e = model(enc_and_dec_states)
    alpha = F.softmax(e, dim=0)
    return (alpha * enc_states).sum(dim=0)

class Encoder(nn.Module):
    def __init__(self, input_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.input_size=input_size
        self.num_hiddens = num_hiddens
        self.rnn = nn.LSTM(input_size, num_hiddens, num_layers,dropout=drop_prob)
        self.linear_x = nn.Linear(input_size, input_size)
        self.linear_h = nn.Linear(num_hiddens, input_size)
        self.linear_att = nn.Linear(input_size, input_size)
    def forward(self, inputs, state):
        # shape:(bs, time_step)
        inputs=inputs.permute(1, 0, 2)
        output=torch.zeros([inputs.shape[0],inputs.shape[1],self.num_hiddens]).to(device)
        att=torch.ones(inputs.shape[1],inputs.shape[2]).to(device)#/inputs.shape[2]
        for i in range(inputs.shape[0]):
            input=inputs[i:i+1,:,:]
            if(state != None):
                x_= self.linear_x(input[0])
                h_ = self.linear_h(state[0][-1])
                att_ = self.linear_att(x_ + h_)
                att = att_.softmax(dim=1)*inputs.shape[2]
            input=input*att
            out,state=self.rnn(input, state)
            output[i]=out
        return output,state

    def begin_state(self):
        return None

class att_combine(nn.Module):
    def __init__(self):
        super(att_combine,self).__init__()
        self.conv_q=nn.Conv1d(1, 128, kernel_size=1, stride=1, padding=0)
        self.conv_k=nn.Conv1d(1, 128, kernel_size=1, stride=1, padding=0)
    def forward(self,qv,k):
        qv_flatten=torch.flatten(qv.permute(1, 0, 2), 1).unsqueeze(1)
        k_flatten=torch.flatten(k.permute(1, 0, 2), 1).unsqueeze(1)
        q_=F.relu(self.conv_q(qv_flatten))
        k_=F.relu(self.conv_k(k_flatten))
        f_=torch.bmm(q_, k_, out=None)
        value=qv_flatten*f_
        value=torch.softmax(value, dim=1).sum(dim=1)
        value=torch.reshape(value,k.shape)
        return value

class Decoder(nn.Module):
    def __init__(self, in_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0,atten=True):
        super(Decoder, self).__init__()
        self.atten=atten
        self.atten_h0= nn.Linear(num_hiddens*2,num_hiddens*2)
        self.atten_h_1 = nn.Linear(num_hiddens, num_hiddens * 2)
        self.atten_hn = nn.Linear(num_hiddens*2, num_hiddens*2)
        self.attention = attention_model(num_hiddens*2, attention_size*2)

        if(self.atten):
            self.rnn = nn.LSTM(num_hiddens*2 + in_size, num_hiddens*2,
                              num_layers, dropout=drop_prob)
        else:
            self.rnn = nn.LSTM(in_size, num_hiddens,
                              num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens*2, 1)

    def forward(self, cur_input, state, enc_states):
        """
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
        if(self.atten):
            c = attention_forward(self.atten_h_1,self.atten_h0, self.atten_hn, self.attention, enc_states, state[0][-1])
            input_and_c = torch.cat((cur_input, c), dim=1)
        else:
            input_and_c=cur_input
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, enc_state):
        # set dec hidden_state as the hidden_state of the last encoder cell
        return enc_state
    def combine_swr(self,dec_state,feature_swr):
        dec_state_out=[]
        dec_state_out.append(torch.cat((dec_state[0], feature_swr), dim=2))
        dec_state_out.append(torch.cat((dec_state[1], feature_swr), dim=2))
        return tuple(dec_state_out)
    def combine_swr_att(self,dec_state,feature_swr):
        dec_state_out=[]
        dec_state_0_att=self.att_combine1(qv=dec_state[0],k=feature_swr)
        feature_swr_0_att=self.att_combine2(qv=feature_swr, k=dec_state[0])
        dec_state_1_att = self.att_combine3(qv=dec_state[1], k=feature_swr)
        feature_swr_1_att = self.att_combine4(qv=feature_swr, k=dec_state[1])
        dec_state_out.append(torch.cat((dec_state_0_att, feature_swr_0_att), dim=2))
        dec_state_out.append(torch.cat((dec_state_1_att, feature_swr_1_att), dim=2))
        return tuple(dec_state_out)

def batch_loss(encoder, decoder, Xa, Xb,loss,feature_swr=None):
    '''
    training process of LSTM with force learning and loss computing
    Xa: historical NWP\LMD\SWR
    Xb: future NWP
    '''
    if(feature_swr is not None):
        feature_swr=feature_swr.permute(1, 0, 2)
    batch_size = Xa.shape[0]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(Xa, enc_state)
    # decoder init state
    dec_state = decoder.begin_state(enc_state)
    if (feature_swr is not None):
        dec_state = decoder.combine_swr(dec_state,feature_swr)
    # decoder infer
    dec_input = torch.zeros([batch_size, Xb.shape[2]]).to(device).float()
    l = torch.tensor([0.0]).to(device)
    for xb in Xb.permute(1,0,2): # Y shape: (batch, seq_len)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + loss(dec_output, xb[:,-1:])
        dec_input = xb  # force learning
    return l

def predict(encoder, decoder,Xa, Xb, output_len,feature_swr=None):
    '''
    predicting process of LSTM with force learning and loss computing
    Xa: historical NWP\LMD\SWR
    Xb: future NWP
    '''
    if (feature_swr is not None):
        feature_swr = feature_swr.permute(1, 0, 2)
    batch_size = Xa.shape[0]
    enc_input = Xa
    enc_state = encoder.begin_state()
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = torch.zeros([batch_size, Xb.shape[2]]).to(device).float()
    dec_state = decoder.begin_state(enc_state)
    if (feature_swr is not None):
        dec_state = decoder.combine_swr(dec_state, feature_swr)
    output = torch.zeros([batch_size,output_len,1]).to((device))
    Xb = Xb.permute(1, 0, 2)
    for i in range(output_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        dec_input = Xb[i]
        dec_input[:, -1:] = dec_output
        output[:,i,:]=dec_output
    return output

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels)/(labels+0.20))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
def masked_mix_loss(preds, labels, null_val=np.nan):
    return masked_mse(preds,labels)+masked_mae(preds,labels)