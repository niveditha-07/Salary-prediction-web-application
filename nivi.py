from matplotlib import pyplot as plt
import streamlit as st #streamlit is used to create web applications
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sal pred", page_icon="")

st.header('SALARY PREDICTION BASED ON EXPERIENCE')

dataset=pd.read_csv('Salary_Data_SLR.csv')
st.write(dataset)
#dataset.columns
X=dataset.iloc[:,0].values
Y=dataset.iloc[:,-1].values
X=X.reshape(-1,1)
#print(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

exp=st.slider('Experience',1,30,1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
#Y_pred=regressor.predict(X_test)
#Y_pred=regressor.predict([[exp]])
#st.write(f"Experience: ", exp)
#st.write(f"Salary: ", float(Y_pred))

if st.button('Predict Salary'):
    Y_pred=regressor.predict(X_test)
    Y_pred=regressor.predict([[exp]])
    st.write(f"Experience: ", exp)
    st.write(f"Salary: ", float(Y_pred))


st.write("""
# Scatter Plot
Salary vs. "Experience"
""")

fig=plt.figure()
plt.scatter(X, Y, alpha=0.8, cmap='viridis')

plt.xlabel('EXPERIENCE')
plt.ylabel('SALARY')
plt.colorbar()

st.pyplot(fig)



    
