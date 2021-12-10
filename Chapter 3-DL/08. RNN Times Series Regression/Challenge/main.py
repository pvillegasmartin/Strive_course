from train import *
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CALLS

input_lang, output_lang, pairs = prepareData('eng', 'spa', True)
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

#Train model

trainIters(pairs, input_lang, output_lang, encoder1, attn_decoder1, 75000, print_every=5000)
torch.save(encoder1.state_dict(), 'encoder1.pth')
torch.save(attn_decoder1.state_dict(), 'attn_decoder1.pth')
'''
#Load model
encoder1.load_state_dict(torch.load('encoder1.pth'))
attn_decoder1.load_state_dict(torch.load('attn_decoder1.pth'))
'''

evaluateRandomly(pairs, input_lang,output_lang,encoder1, attn_decoder1)

#evaluateAndShowAttention("elle a cinq ans de moins que moi .")

