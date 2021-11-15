import pandas as pd
import numpy as np
import math
import streamlit as st
from matplotlib import pyplot
from matplotlib import rcParams

def var_pythonic(array):
    """ Calculates the variance of an array that contains values of a sample of a
    population.

    Arguments
    ---------
    array : array, contains sample of values.

    Returns
    -------
    var   : float, variance of the array .
    """
    var = np.sum((array - array.mean()) ** 2) / (len(array) - 1)
    return var


def sample_std(array):
    """ Computes the standard deviation of an array that contains values
    of a sample of a population.

    Arguments
    ---------
    array : array, contains sample of values.

    Returns
    -------
    std   : float, standard deviation of the array.
    """

    std = math.sqrt(var_pythonic(array))
    return std


def std_percentages(x, x_mean, x_std):
    """ Computes the percentage of coverage at 1std, 2std and 3std from the
    mean value of a certain variable x.

    Arguments
    ---------
    x      : array, data we want to compute on.
    x_mean : float, mean value of x array.
    x_std  : float, standard deviation of x array.

    Returns
    -------

    per_std_1 : float, percentage of values within 1 standard deviation.
    per_std_2 : float, percentage of values within 2 standard deviations.
    per_std_3 : float, percentage of values within 3 standard deviations.
    """

    std_1 = x_std
    std_2 = 2 * x_std
    std_3 = 3 * x_std

    elem_std_1 = np.logical_and((x_mean - std_1) < x, x < (x_mean + std_1)).sum()
    per_std_1 = elem_std_1 * 100 / len(x)

    elem_std_2 = np.logical_and((x_mean - std_2) < x, x < (x_mean + std_2)).sum()
    per_std_2 = elem_std_2 * 100 / len(x)

    elem_std_3 = np.logical_and((x_mean - std_3) < x, x < (x_mean + std_3)).sum()
    per_std_3 = elem_std_3 * 100 / len(x)

    return per_std_1, per_std_2, per_std_3

def stream_view(figure1,figure2,abv_std1_per, abv_std2_per, abv_std3_per,ibu_std1_per, ibu_std2_per, ibu_std3_per):
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;font-size:55px;'>Beers exercise</h1>", unsafe_allow_html=True)
    st.markdown("Every time that we work with data, visualizing it is very useful. Visualizations give us a better idea of how our data behaves. One way of visualizing data is with a frequency-distribution plot known as **histogram**: a graphical representation of how the data is distributed. To make a histogram, first we need to 'bin' the range of values (divide the range into intervals) and then we count how many data values fall into each interval. The intervals are usually consecutive (not always), of equal size and non-overlapping.")
    col1, col2 = st.columns((1,1))
    col1.pyplot(figure1)
    col2.pyplot(figure2)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<h1 style='text-align: center;font-size:55px;'>Comparing with a normal distribution</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("A **normal** (or Gaussian) distribution is a special type of distrubution that behaves as shown in the figure: 68% of the values are within one standard deviation $\sigma$ from the mean; 95% lie within $2\sigma$; and at a distance of $\pm3\sigma$ from the mean, we cover 99.7% of the values. This fact is known as the $3$-$\sigma$ rule, or 68-95-99.7 (empirical) rule.")
    col1, col2 = st.columns((1, 1))
    col1.markdown("## ABV")
    col1.markdown(f"""<p style='text-align: center;'>The percentage of coverage at 1 std of the abv_mean is : {round(abv_std1_per,2)} %</p>""", unsafe_allow_html=True)
    col1.markdown(f"""<p style='text-align: center;'>The percentage of coverage at 2 std of the abv_mean is : {round(abv_std2_per,2)} % </p>""", unsafe_allow_html=True)
    col1.markdown(f"""<p style='text-align: center;'>The percentage of coverage at 3 std of the abv_mean is : {round(abv_std3_per,2)} % </p>""", unsafe_allow_html=True)

    col2.markdown("## IBU")
    col2.markdown(
        f"""<p style='text-align: center;'>The percentage of coverage at 1 std of the abv_mean is : {round(ibu_std1_per, 2)} %</p>""",
        unsafe_allow_html=True)
    col2.markdown(
        f"""<p style='text-align: center;'>The percentage of coverage at 2 std of the abv_mean is : {round(ibu_std2_per, 2)} % </p>""",
        unsafe_allow_html=True)
    col2.markdown(
        f"""<p style='text-align: center;'>The percentage of coverage at 3 std of the abv_mean is : {round(ibu_std3_per, 2)} % </p>""",
        unsafe_allow_html=True)

if __name__=="__main__":
    #read data
    from urllib.request import urlretrieve
    URL = 'http://go.gwu.edu/engcomp2data1'
    urlretrieve(URL, './data/beers.csv')
    beers = pd.read_csv("data/beers.csv")

    ibu_clean = beers['ibu'].dropna()
    abv_clean = beers['abv'].dropna()

    # Set font style and size
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 16
    # You can set the size of the figure by doing:
    figure1 = pyplot.figure(figsize=(10, 5))

    # Plotting
    pyplot.hist(abv_clean, bins=20, color='#3498db', histtype='bar', edgecolor='white')
    # The \n is to leave a blank line between the title and the plot
    pyplot.title('abv \n')
    pyplot.xlabel('Alcohol by Volume (abv) ')
    pyplot.ylabel('Frequency');

    # You can set the size of the figure by doing:
    figure2 = pyplot.figure(figsize=(10, 5))

    # Plotting
    pyplot.hist(ibu_clean, bins=20, color='#e67e22', histtype='bar', edgecolor='white')
    # The \n is to leave a blanck line between the title and the plot
    pyplot.title('ibu \n')
    pyplot.xlabel('International Bittering Units (ibu)')
    pyplot.ylabel('Frequency');

    abv_std = sample_std(abv_clean)
    ibu_std = sample_std(ibu_clean)

    abv_mean = np.mean(abv_clean)
    ibu_mean = np.mean(ibu_clean)

    abv_std1_per, abv_std2_per, abv_std3_per = std_percentages(abv_clean, abv_mean, abv_std)
    ibu_std1_per, ibu_std2_per, ibu_std3_per = std_percentages(ibu_clean, ibu_mean, ibu_std)

    stream_view(figure1,figure2,abv_std1_per, abv_std2_per, abv_std3_per,ibu_std1_per, ibu_std2_per, ibu_std3_per)