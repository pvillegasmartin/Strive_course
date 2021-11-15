import argparse
import pickle
import numpy as np

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

parser = argparse.ArgumentParser(allow_abbrev=False)

parser = argparse.ArgumentParser(description='''Aelius Galan: This progam says if the patient has real chances to suffer a heart attack. (It has a 88.16% accuracy.)''')

parser.add_argument("age", type=int, help="Age of the person")
parser.add_argument("sex", type=int,choices=[0,1], help="Gender of the person")
parser.add_argument("cp", type=int,choices=[0,1,2,3], help="Chest Pain type")
parser.add_argument("trtbps", type=int, help="Resting blood pressure (in mm Hg)")
parser.add_argument("chol", type=int, help="Cholestoral in mg/dl fetched via BMI sensor")
parser.add_argument("fbs", type=int,choices=[0,1], help="(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)")
parser.add_argument("restecg", type=int,choices=[0,1,2], help="Resting electrocardiographic results")
parser.add_argument("thalachh", type=int, help="Maximum heart rate achieved")
parser.add_argument("exng", type=int,choices=[0,1], help="Exercise induced angina (1 = yes; 0 = no)")
parser.add_argument("oldpeak", type=float, help="Previous peak")
parser.add_argument("slp", type=int,choices=[0,1,2], help="Slope")
parser.add_argument("caa", type=int,choices=[0,1,2,3], help="Number of major vessels (0-3)")
parser.add_argument("thall", type=int,choices=[0,1,2,3], help="Thal rate")


args = parser.parse_args()


answer = loaded_model.predict(np.array([args.age,args.sex,args.cp,args.trtbps,args.chol,args.fbs,\
                                       args.restecg, args.thalachh, args.exng, args.oldpeak, args.slp, args.caa, args.thall, args.slp*args.oldpeak] ).reshape(1,14))
if answer:
    print(f'This person have high chances of suffer a heart attack')
else:
    print(f"This person probably won't suffer a heart attack")


