from flask import Flask, render_template
app = Flask(__name__)

# ---- ABOUT ME ----
personal_description = "I am a Spanish guy born in 1993 who is enthusiastic about numbers. Since I was little I had a special affection to the numbers and that facilitated my learning of all the subjects related to it. After years of training and work experience in which I have acquired knowledge and above all other more general skills, I consider that I am ready to fully focus my career on data analysis. What is life if not a succession of decision making? Let's take them with the greatest and best information possible"

skills = [{'id':'1', 'name':'Maths and Statistics', 'description':'Advanced knowledge acquired mainly in my university studies and complemented with a series of online courses.'},
          {'id':'2', 'name':'Python', 'description':'From the first computer science course I took at university, it fascinated me and made me change the focus of my professional career. I have mostly learned it on my own with the help of online courses and tutorials.'},
          {'id':'3', 'name':'Analytics', 'description':'I am becoming more deterministic, good data analysis helps us make better decisions. I have been working on personal projects for years and finally this year I decided to focus my efforts on expanding my knowledge related to data analysis to focus my professional career on it.'},
          {'id':'4', 'name':'Data visualization', 'description':'Knowing how to read the data is just as important as being able to explain it. In addition to knowing how to use different libraries and applications on python to display data in a useful, friendly and easy way, I have advanced knowledge in Power BI.'},
          {'id':'5', 'name':'Manegement', 'description':'One of the acquired soft characteristics that I value most from my experience working in consulting has been the organization, not only at the team level but also at the personal level, knowing how to work using the agile methodology.'}]

# ---- PROJECTS ----
ds_projects = [{'title':'Book recommender',
                    'img':'/static/libros.JPG',
                    'img_desc':'Book_recommender_image',
                    'text':'Basic project of an author/book recommender.',
                    'technologies':[('Python', '/static/python.PNG'),('Streamlit', '/static/streamlit.PNG')],
                    'date':'15-10-2021',
                    'link':'https://github.com/pvillegasmartin/challange_week_1'},
                {'title':'Activity tracker',
                    'img':'/static/heartbeat.PNG',
                    'img_desc':'Heartbeat_black_and_white',
                    'text':"Application in kivy with a machine learning algorithm behind to follow a person's activity",
                    'technologies':[('Python', '/static/python.PNG'),('Kivy', '/static/kivy.PNG')],
                    'date':'12-11-2021',
                    'link':'https://github.com/pvillegasmartin/Challenge-week-2'}
    ]
pbi_projects = []
other_projects = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about_me')
def about_me():
    return render_template('about_me.html', skills=skills, personal_description=personal_description)

@app.route('/my_projects')
def my_projects():
    return render_template('my_projects.html', ds_projects = ds_projects, pbi_projects = pbi_projects, other_projects = other_projects)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)