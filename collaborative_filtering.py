import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


##### The del operation was intended to free ram inturn optimizing runtime and memory occupancy####

##### Read and manipulate to desired form   ####

df = pd.read_csv("netflix/TrainingRatings.txt", sep=",", header=None)
df.columns = ["movies", "users", "ratings"]
users = np.sort(pd.unique(df["users"]))
movies = np.sort(pd.unique(df["movies"]))

df = df.pivot(index='users', columns='movies', values='ratings')

df.astype('float32')  # To reduce the memory occupied bu the data

#
#### Calculate Mean for each user  ########

mean = df.mean(1)
# print("Mean = ", mean)
cccc = (df.loc[:, 8]).to_numpy()
centered_at_zero = df.sub(mean, axis="index")
del df


# Centering the ratings at zero

centered_at_zero = centered_at_zero.fillna(0)

vi = centered_at_zero.to_numpy()
vi.astype(np.float16)

del centered_at_zero

denominator = np.linalg.norm(vi, axis=1)

###### Calculate weight for each user at a time #####
# the numerator and denominator are calculated simulteniously to save running time

weight = []
for d, n, i in zip(denominator, vi, range(28978)):

    # print(i)
    dot_product_deno = np.multiply(d, denominator)
    dot_product_deno = np.where(dot_product_deno == 0, 1, dot_product_deno)
    dot_product_neu = np.dot(n, np.transpose(vi))

    weight.append(np.divide(dot_product_neu, dot_product_deno))

weight = np.asarray(weight)
print(weight)


##### Making the user's owin with it's self to zero####

np.fill_diagonal(weight, 0)


##### filtering Neighbours Based on similarity rather taking a fixed ammount of simmalar users####

weight = np.where(weight > 0.196, weight, 0)
# print("0.191")

kappa = np.divide(1, (np.sum(np.absolute(weight), axis=1)))
np.nan_to_num(kappa, 0)
print("kappa", kappa)


# ####### Calculate prediction#####

# # ##### Calculating prediction for every user movie combination rather then test######
### filtering users and movies was observed to take more time than calculating all the weights###

temp = np.dot(weight, vi)
prediction_offset = np.multiply(kappa, temp.T).T

va = mean.to_numpy()

prediction = np.add(va, prediction_offset.T).T

# # # ####### Rounding the predicted reviews#####
prediction = np.round(prediction)


df1 = pd.read_csv('netflix/TestingRatings.txt',
                  sep=',', header=None, dtype=int)
x = df1.to_numpy()

y_pred = []


###### Taking out the users who's corresponding movie rating is required ####

for d in x:

    user_id = np.where(users == d[1])
    movie_id = np.where(movies == d[0])

    temppp = prediction[user_id, movie_id]

    tt = temppp[0]
    y_pred.append(tt[0])

y_pred = np.asarray(y_pred)
# print(y_pred)
y_true = x[:, 2]
# print(y_true)
rmse = mean_squared_error(y_true, y_pred)
print(rmse)
mas = mean_absolute_error(y_true, y_pred)
print(mas)
print(np.amax(weight))
print(np.amin(weight))
