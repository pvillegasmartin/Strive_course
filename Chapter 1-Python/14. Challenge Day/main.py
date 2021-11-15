
from matplotlib.pyplot import subplot
from matplotlib import animation
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np, math, matplotlib.patches as patches



digits = load_digits()
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
digits_df = pd.DataFrame({"x":proj[:, 0], "y":proj[:, 1], "label":digits.target})
df_numbers = digits_df.groupby('label').mean()
df_numbers['label']=df_numbers.index

#Graph 1 point per label
fig_numbers = plt.figure(figsize=(5,3))
ax = fig_numbers.add_subplot(1,1,1)
plt.xlim(-40,40)
plt.xlabel("x")
plt.ylabel("y")
ylabels = range(-15,25,5)
plt.yticks(ylabels)
for i in df_numbers.index.values.tolist():
    plt.scatter(df_numbers.loc[i:i,'x'],df_numbers.loc[i:i,'y'], marker='o', label=i)
plt.legend()

#IMPROVEMENT
df_dist = pd.DataFrame(columns=['filter','from','dist'])
print(df_dist)
for i in df_numbers.index.values.tolist():
    for j in df_numbers.index.values.tolist():
        df_dist = df_dist.append({'filter':int(i),'from':int(j),'dist':np.sqrt((df_numbers.loc[i,'x']-df_numbers.loc[j,'x'])**2 + (df_numbers.loc[i,'y']-df_numbers.loc[j,'y'])**2)}, ignore_index=True)
        df_dist['filter'] = df_dist['filter'].apply(np.int64)
        df_dist['from'] = df_dist['from'].apply(np.int64)




###########################
sayid = st.container()
vector_plot = st.container()
BMI_plot = st.container()
pablos = st.container()



with sayid:
    st.title('Challenge Project')

    # this container contains items belong the vector exercise
    with vector_plot: 
        st.header('transformation of vector')

        #creates two columns, one for user input, one for displaying the plot
        vec_inputs,vec_disp = st.columns(2) 

        #takes user input for angle
        angle = vec_inputs.slider('chose angle of transformation(in degrees)',min_value=0,max_value=360,value=30,step=10) 
        vector_x1 = float(vec_inputs.text_input('input value for x cordinate of the vector','1'))
        vector_y1 = float(vec_inputs.text_input('input value for y cordinate of the vector','1'))

        #function returns transformed vector
        def vec_plotter(vec,theta):
            """
            Function takes two variables , vector and theta as input for rotation matrix.
            Then it creates rotatition matrix, with variable name as "rot_mat"
            it saves the result vector to the variable of "result"
            """
            rot_mat = np.array([[np.cos(theta),-np.sin(theta)],
                                [np.sin(theta),np.cos(theta)]])
            result = np.dot(rot_mat,vec)

            return result
        ###############




# Create figure
        fig = plt.figure()
        ax = fig.gca()

# Axes labels and title are established
        ax = fig.gca()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_ylim(-2*vector_x1, 2*vector_x1)
        ax.set_xlim(-2*vector_x1, 2*vector_x1)
        plt.gca().set_aspect('equal', adjustable='box')

        # x = np.linspace(-1, 1, 20)
        # y = np.linspace(-1, 1, 20)
        # dx = np.zeros(len(x))
        # dy = np.zeros(len(y))

        # for i in range(len(x)):
        #     dx[i] = math.sin(x[i])
        #     dy[i] = math.cos(y[i])

        # patch = patches.Arrow(0, 0, vector_x1, vector_y1)
        # angle_new=angle*np.pi/180
        # magnitude = np.sqrt(vector_x1**2 + vector_y1**2)
        # angles = np.linspace(np.arctan(vector_y1/vector_x1),np.arctan(vector_y1/vector_x1)+angle_new,200)
        # def init():
        #     ax.add_patch(patch)
        #     return patch,


        # def animate(t):
        #     global patch

        #     ax.patches.remove(patch)

        #     patch = plt.Arrow(0, 0, magnitude*np.cos(angles[t]),magnitude*np.sin(angles[t]))
        #     ax.add_patch(patch)

        #     return patch,


        # anim = animation.FuncAnimation(fig, animate,
        #                             init_func=init,
        #                             interval=20,
        #                             blit=False)

        # st.pyplot(fig)




        ##############

        #code below, displays the plot on the browser
        vec = [vector_x1,vector_y1]
        result = vec_plotter(vec,(angle/180)*np.pi)
        fig,ax = plt.subplots(figsize=(10,8))
        ax.arrow(0,0,vec[0],vec[1],head_width=0.05,head_length=0.1,lw=3,fc='green',ec='green',label='original') #plots original in green
        ax.arrow(0,0,result[0],result[1],head_width=0.05,head_length=0.1,lw=3,fc='blue',ec='blue',label='transformed')#plots transformed in blue
        plt.xlim(-2*abs(vector_x1),2*abs(vector_x1))
        plt.ylim(-2*abs(vector_x1),2*abs(vector_x1))
        plt.legend(loc='lower right')
        plt.grid()

        vec_disp.write(fig)#displays the plot

    
    #container BMI_plot contains items related to the population data visualisation
    with BMI_plot:
        st.header('BMI index of given population')
        st.write('\n'*5)
        
        url ="https://raw.githubusercontent.com/chriswmann/datasets/master/500_Person_Gender_Height_Weight_Index.csv"
        
        @st.cache  #this line of code, helps us to upload our data fast, basically unless we change url of the file, it saves it in the cache, so file will be uploaded once
        def get_file(url):
            df = pd.read_csv(url)
            return df
       
        df = get_file(url) #getting the file via url

        
        st.subheader('here you can glance at row data')

        #showing first 100 elements of raw data
        st.write(df.head(100))
        st.header("let's play with data visualization")
        user_input, disp = st.columns(2)

        #takes user input to indentify the type of plot he/she wants
        opts = ['scatter','historgam']
        opt = user_input.radio('what kind of plot you want to visualise population data:',opts)
        
        ###################### S C A T T E R ###################################
        if opt == 'scatter':
            check = user_input.checkbox("show mean")
            who = user_input.selectbox('what gender you want to choose',('Male','Female','All'))
            fig,ax = plt.subplots()
            if who == 'All':
                ax.scatter(df['Weight'],df['Height'],c=df.Index)
                if check:
                    ax.scatter(df['Weight'].mean(),df['Height'].mean(),c='red',marker='x')
                    ax.text(df['Weight'].mean()-10,df['Height'].mean()-3,'mean value',color='red')

                ax.set_title('population\'s BMI index scatter plot')
                ax.set_xlabel('weight [pounds]')
                ax.set_ylabel('height [centimeters]')
                disp.write(fig)
            else:
                df_new = df[df["Gender"]==who]
                ax.scatter(df_new['Weight'],df_new['Height'],c=df_new.Index)
                if check:
                    ax.scatter(df_new['Weight'].mean(),df_new['Height'].mean(),c='red',marker='x')
                    ax.text(df_new['Weight'].mean()-10,df_new['Height'].mean()-3,'mean value',color='red')

                ax.set_title('{} BMI index scatter plot'.format(who))
                ax.set_xlabel('weight [pounds]')
                ax.set_ylabel('height [centimeters]')
                disp.write(fig)
            
        ###################### H I S T O G R A M ###################################
        else:
            who = user_input.selectbox('what gender you want to choose',('Male','Female','All'))
            fig,ax = plt.subplots()
            what = user_input.selectbox('what kind parametr you to choose',('Weight','Height'))

            if who == 'All':
                 bin = user_input.slider('how many bins you want in histogram',min_value=2,max_value=len(df[what]),step=1,value=20)
                 ax.hist(df[what],bins=bin,ec='white')
                 ax.set_title('{} of given population'.format(what))
                 ax.grid()
                 ax.set_xlabel('{}'.format(what))
                 ax.set_ylabel('frequency')
                 disp.write(fig)
            else:
                df_new = df[df["Gender"]==who]
                bin = user_input.slider('how many bins you want in histogram',min_value=2,max_value=len(df_new[what]),step=1,value=20)
                ax.hist(df_new[what],bins=bin,ec='white')
                ax.set_title('{}\'s {}  histogram plot'.format(who,what))
                ax.set_xlabel('{}'.format(what))
                ax.set_ylabel('frequency')
                ax.grid()
                disp.write(fig)


with pablos:
    st.title('PCA exercise')
    col1, col2 = st.columns((2,1.5))
    col1.subheader("mean of PCA coordinates")
    col1.pyplot(fig_numbers)
    col2.subheader("list of similarity")
    number_studied = col2.selectbox('Select a number',df_numbers['label'].unique())
    data_order_numbers = pd.Series(df_dist[(df_dist['filter'] == number_studied) & (df_dist['from'] != number_studied)].sort_values(by='dist', ascending=True)['from'])
    col2.write(data_order_numbers.values)