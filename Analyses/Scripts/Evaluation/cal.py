import matplotlib.pyplot as plt

# Data for the bar chart
countries = ['US', 'UK']
total_population = [234623, 234711]
polarized_users = [185723, 180391]

# Calculate non-polarized users
non_polarized_users = [total_population[i] - polarized_users[i] for i in range(len(total_population))]

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 7))

bar_width = 0.35
index = range(len(countries))

# Bars for polarized and non-polarized users
bar1 = ax.bar(index, non_polarized_users, bar_width, label='Non-polarized Users', color='#66b3ff')
bar2 = ax.bar([i + bar_width for i in index], polarized_users, bar_width, label='Polarized Users', color='#ffcc99')

# Labels and title
ax.set_xlabel('Country', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Users', fontsize=14, fontweight='bold')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(countries, fontsize=12)

# Legend
ax.legend()

# Display the bar chart
plt.tight_layout()
plt.show()
