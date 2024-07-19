import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the preprocessed user data file
user_data = pd.read_csv('Analyses/User Data/Collected Stances/usersUK>1_stances.csv')

# Calculate the number of opinions each user has
opinion_columns = [
    'pro_brexit', 'anti_brexit',
    'pro_climateAction', 'anti_climateAction',
    'pro_NHS', 'anti_NHS',
    'pro_israel', 'pro_palestine',
    'pro_company_taxation', 'pro_worker_taxation',
    'Brexit_neutral', 'ClimateChangeUK_neutral',
    'HealthcareUK_neutral', 'IsraelPalestineUK_neutral',
    'TaxationUK_neutral'
]

# opinion_columns = [
#     'pro_immigration', 'anti_immigration',
#     'pro_climateAction', 'anti_climateAction',
#     'public_healthcare', 'private_healthcare',
#     'pro_israel', 'pro_palestine',
#     'pro_middle_low_tax', 'pro_wealthy_corpo_tax',
#     'ImmigrationUS_neutral', 'ClimateChangeUS_neutral',
#     'HealthcareUS_neutral', 'IsraelPalestineUS_neutral',
#     'TaxationUS_neutral',
# ]

user_data['total_opinions'] = user_data[opinion_columns].sum(axis=1)

# Generate the normal distribution of the users based on the number of opinions
mu, std = norm.fit(user_data['total_opinions'])

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(user_data['total_opinions'], bins=30, density=True, alpha=0.6, color='b')

# Plot the PDF
xmin, xmax = -200, 200  # Adjusted x-axis range
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

title = f"Fit results: mu = {mu:.2f},  std = {std:.2f}"
plt.title(title)
plt.xlabel('Number of Opinions')
plt.ylabel('Density')
plt.xlim(xmin, xmax)  # Set the x-axis limits
plt.grid(True)
plt.show()

# Print key values of the distribution
print(f"Mean (mu): {mu}")
print(f"Standard Deviation (std): {std}")
print(f"1 standard deviation range: {mu - std} to {mu + std}")
print(f"2 standard deviations range: {mu - 2 * std} to {mu + 2 * std}")

# Filter users within 2 standard deviations
lower_bound = mu - 2 * std
upper_bound = mu + 2 * std
filtered_users = user_data[(user_data['total_opinions'] >= lower_bound) & (user_data['total_opinions'] <= upper_bound)]

# Print the number of users to be deleted
num_users_to_delete = len(user_data) - len(filtered_users)
print(f"Number of users to be deleted: {num_users_to_delete}")

# Prompt for user confirmation
confirm = input("Do you want to delete these users and save the cleaned dataset? (y/n): ").strip().lower()
if confirm == 'y':
    filtered_users.to_csv('Analyses/User Data/Preprocessed/usersUK>1_preprocessed.csv', index=False)
    print("Cleaned dataset saved as 'usersUK>1_preprocessed.csv'.")
else:
    print("No changes made.")
