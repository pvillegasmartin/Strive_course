from translator import *
import streamlit as st
from parameters import *

@st.cache(suppress_st_warning=True)
def streamlit_template(encoder_es, encoder_se,attn_decoder_es,attn_decoder_se, input_lang_se, output_lang_se, input_lang_es, output_lang_es):

    st.sidebar.markdown("<h1 style=' color: #948888;'>TRANSLATOR</h1>",
                        unsafe_allow_html=True)
    st.sidebar.write('\n')
    analysis_type = st.sidebar.radio("What translation do you need?", ('Spanish-English', 'English-Spanish'))

    if analysis_type == 'Spanish-English':
        st.title("TRADUCTOR: Español - Ingles")

        user_input = st.text_input("Tu frase..")
        traducir = st.button(f"¡Traducir!")
        if user_input and traducir:
            traduccion = evaluateAndShowAttention(input_lang_se, output_lang_se, encoder_se, attn_decoder_se, user_input)
            st.markdown(f"<h2 style=' color: #948888;'>{traduccion}</h2>", unsafe_allow_html=True)

    elif analysis_type == 'English-Spanish':
        st.title("TRANSLATOR: English - Spanish")
        user_input = st.text_input("Your sentence..")
        traducir = st.button(f"¡Translate!")
        if user_input and traducir:
            traduccion = evaluateAndShowAttention(input_lang_es, output_lang_es, encoder_es, attn_decoder_es, user_input)
            st.markdown(f"<h2 style=' color: #948888;'>{traduccion}</h2>", unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    input_lang_se, output_lang_se, pairs_se = prepareData('eng', 'spa', True)
    input_lang_es, output_lang_es, pairs_es = prepareData('eng', 'spa', False)

    encoder_es = EncoderRNN(input_lang_es.n_words, hidden_size).to(device)
    encoder_se = EncoderRNN(input_lang_se.n_words, hidden_size).to(device)
    attn_decoder_se = AttnDecoderRNN(hidden_size, output_lang_se.n_words, dropout_p=0.1).to(device)
    attn_decoder_es = AttnDecoderRNN(hidden_size, output_lang_es.n_words, dropout_p=0.1).to(device)
    encoder_es.load_state_dict(torch.load('encoder_eng-spa.pth'))
    attn_decoder_es.load_state_dict(torch.load('attn_decoder_eng-spa.pth'))
    encoder_se.load_state_dict(torch.load('encoder_spa-eng.pth'))
    attn_decoder_se.load_state_dict(torch.load('attn_decoder_spa-eng.pth'))

    streamlit_template(encoder_es, encoder_se,attn_decoder_es,attn_decoder_se, input_lang_se, output_lang_se, input_lang_es, output_lang_es)