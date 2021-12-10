from utils import *
import streamlit as st


def streamlit_template(input_lang_se, output_lang_se, pairs_se, input_lang_es, output_lang_es, pairs_es):
    st.set_page_config(layout="wide")
    st.sidebar.markdown("<h1 style=' color: #948888;'>TRANSLATOR</h1>",
                        unsafe_allow_html=True)
    st.sidebar.write('\n')
    analysis_type = st.sidebar.radio("What translation do you need?", ('Spanish-English', 'English-Spanish'))

    if analysis_type == 'Spanish-English':
        st.title("TRADUCTOR: Español - Ingles")

        # Charge model
        input_lang, output_lang, pairs = input_lang_se, output_lang_se, pairs_se
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
        encoder1.load_state_dict(torch.load('encoder_spa-eng.pth'))
        attn_decoder1.load_state_dict(torch.load('attn_decoder_spa-eng.pth'))

        user_input = st.text_input("Tu frase..")
        traducir = st.button(f"¡Traducir!")
        if user_input and traducir:
            traduccion = evaluateAndShowAttention(input_lang, output_lang, encoder1, attn_decoder1, user_input)
            st.markdown(f"<h2 style=' color: #948888;'>{traduccion}</h2>", unsafe_allow_html=True)

    elif analysis_type == 'English-Spanish':
        st.title("TRANSLATOR: English - Spanish")

        # Charge model
        input_lang, output_lang, pairs = input_lang_es, output_lang_es, pairs_es
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
        encoder1.load_state_dict(torch.load('encoder_eng-spa.pth'))
        attn_decoder1.load_state_dict(torch.load('attn_decoder_eng-spa.pth'))

        user_input = st.text_input("Your sentence..")
        traducir = st.button(f"¡Translate!")
        if user_input and traducir:
            traduccion = evaluateAndShowAttention(input_lang, output_lang, encoder1, attn_decoder1, user_input)
            st.markdown(f"<h2 style=' color: #948888;'>{traduccion}</h2>", unsafe_allow_html=True)


if __name__ == "__main__":
    input_lang_se, output_lang_se, pairs_se = prepareData('eng', 'spa', True)
    input_lang_es, output_lang_es, pairs_es = prepareData('eng', 'spa', False)
    streamlit_template(input_lang_se, output_lang_se, pairs_se, input_lang_es, output_lang_es, pairs_es)