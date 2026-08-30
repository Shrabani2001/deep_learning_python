import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Read CSV Dataset
df = pd.read_csv("students.csv")
print(df)

# 2. Separate Input and Output

#Input features
x = df[["study_hours", "attendance"]]

#Target (output)
y = df["result"]

# Convert to numpy arrays because input and output are the numeric value
x = x.values
y = y.values

# 3. Split Dataset into Training and Testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n============= DATA SPLIT ==============")
print("Training Data:", len(x_train))
print("Testing Data:", len(x_test))

# 4. Feature Scaling
scalar = StandardScaler()

x_train = scalar.fit_transform(x_train) # fit_transform() Calculate the mean and the standard deviation of the training data
x_test = scalar.transform(x_test) #transform() going to use those values to scale the training data

#Save scaler for future prediction
joblib.dump(scalar, "student_scaler.pkl") # new data scaled before giving to model

print("\n============= DATA SCALING ==============")
print("Training data scaled successfully")


# 5. Create Neural Network

model = tf.keras.Sequential([

    # hidden layer  8 is the number of neurons
    tf.keras.layers.Dense(
        8,
        activation="relu",
        input_shape=(2,)
    ),
    # output layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    ) #sigmoid going to present the result in binary 0 or 1 format
])

# 6. Compile Model
model.compile(
    optimizer="adam", #most fastest optimizer
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 7. Display Model Structure
print("\n============= MODEL STRUCTURE ==============")
model.summary()

# 8. Train the Neural Network

print("\n============= MODEL TRAINING ==============")
model.fit(x_train, y_train, epochs=20)

# 9. Evaluate Model Performance

loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# 10. Save Trained Model to a File

model.save("student_pass_fail_ann.keras")

print("\n============= MODEL SAVED ==============")
print("Model saved as: student_pass_fail_ann.keras")