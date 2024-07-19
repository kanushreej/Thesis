import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.cluster import KMeans

# Define major political events
events = {
    'Brexit Referendum': datetime(2016, 6, 23),
    'US Presidential Election 2016': datetime(2016, 11, 8),
    'Paris Climate Agreement Adoption': datetime(2015, 12, 12),
    'US Withdrawal from Paris Agreement': datetime(2020, 11, 4),
    'US Supreme Court Ruling on ACA (2015)': datetime(2015, 6, 25)
}

# Define time ranges around events
def get_time_ranges(event_date):
    before_start = event_date - timedelta(days=90)
    before_end = event_date
    during_start = event_date
    during_end = event_date + timedelta(days=30)
    after_start = event_date + timedelta(days=30)
    after_end = event_date + timedelta(days=120)
    return before_start, before_end, during_start, during_end, after_start, after_end

# Topic files
topic_files = {
    'Brexit': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/UK/Brexit_labelled.csv',
    'ClimateChangeUK': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/UK/ClimateChangeUK_labelled.csv',
    'HealthcareUK': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/UK//HealthcareUK_labelled.csv',
    'IsraelPalestineUK': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/UK/IsraelPalestineUK_labelled.csv',
    'TaxationUK': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/UK/TaxationUK_labelled.csv'
}

# Opinion columns for different topics
opinion_columns = {
    'Brexit': ['pro_brexit', 'anti_brexit'],
    'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
    'HealthcareUK': ['pro_NHS', 'anti_NHS'],
    'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
    'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation']
}

# Generate time ranges for all events
time_ranges = {event: get_time_ranges(date) for event, date in events.items()}

def filter_and_cross_reference(event_name, time_ranges, topic_files):
    before_start, before_end, during_start, during_end, after_start, after_end = time_ranges[event_name]
    
    # Convert all ranges to timezone-naive datetime objects
    before_start, before_end = before_start.replace(tzinfo=None), before_end.replace(tzinfo=None)
    during_start, during_end = during_start.replace(tzinfo=None), during_end.replace(tzinfo=None)
    after_start, after_end = after_start.replace(tzinfo=None), after_end.replace(tzinfo=None)
    
    df_before_list = []
    df_during_list = []
    df_after_list = []

    for topic, file_path in topic_files.items():
        df = pd.read_csv(file_path)
        df['created_utc'] = pd.to_datetime(df['created_utc']).dt.tz_localize(None)
        
        if not all(col in df.columns for col in opinion_columns[topic]):
            print(f"Missing columns in {file_path}: {set(opinion_columns[topic]) - set(df.columns)}")
            continue

        df_before = df[(df['created_utc'] >= before_start) & (df['created_utc'] < before_end)]
        df_during = df[(df['created_utc'] >= during_start) & (df['created_utc'] < during_end)]
        df_after = df[(df['created_utc'] >= after_start) & (df['created_utc'] < after_end)]

        df_before_filtered = df_before[(df[opinion_columns[topic][0]] != 0) | (df[opinion_columns[topic][1]] != 0)]
        df_during_filtered = df_during[(df[opinion_columns[topic][0]] != 0) | (df[opinion_columns[topic][1]] != 0)]
        df_after_filtered = df_after[(df[opinion_columns[topic][0]] != 0) | (df[opinion_columns[topic][1]] != 0)]

        df_before_list.append(df_before_filtered)
        df_during_list.append(df_during_filtered)
        df_after_list.append(df_after_filtered)

    if not df_before_list or not df_during_list or not df_after_list:
        print(f"No data available for event: {event_name}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    df_before = pd.concat(df_before_list, ignore_index=True)
    df_during = pd.concat(df_during_list, ignore_index=True)
    df_after = pd.concat(df_after_list, ignore_index=True)

    return df_before, df_during, df_after

def aggregate_stances(time_ranges, topic_files):
    df_before_list = []
    df_during_list = []
    df_after_list = []

    for event_name in time_ranges.keys():
        before, during, after = filter_and_cross_reference(event_name, time_ranges, topic_files)
        if not before.empty:
            df_before_list.append(before)
        if not during.empty:
            df_during_list.append(during)
        if not after.empty:
            df_after_list.append(after)

    df_before = pd.concat(df_before_list, ignore_index=True) if df_before_list else pd.DataFrame()
    df_during = pd.concat(df_during_list, ignore_index=True) if df_during_list else pd.DataFrame()
    df_after = pd.concat(df_after_list, ignore_index=True) if df_after_list else pd.DataFrame()

    return df_before, df_during, df_after

# Aggregate stances
df_before, df_during, df_after = aggregate_stances(time_ranges, topic_files)

# Normalize the opinion columns for clustering
all_opinion_columns = sum(opinion_columns.values(), [])

def perform_clustering_with_BERTopic(df, period, event_name):
    if df.empty:
        print(f"No data to cluster for {event_name} during {period} period")
        return df

    # Handle NaNs by filling with zero
    df = df.fillna(0)

    # Normalize the opinion columns
    scaler = StandardScaler()
    df[all_opinion_columns] = scaler.fit_transform(df[all_opinion_columns])

    # Convert normalized opinion columns to a string format for BERTopic
    df['opinions'] = df[all_opinion_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # Create a CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['opinions'])

    # Create and fit BERTopic model
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(df['opinions'].tolist())  # Pass list of strings to BERTopic

    # Add topics to DataFrame
    df['topic'] = topics

    # Extract topic representations
    topic_representations = topic_model.get_topic_info()

    # Apply KMeans clustering on the topic representations
    num_clusters = len(topic_representations)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(probs)
    df['kmeans_cluster'] = kmeans.labels_

    # Save the clustered data to a CSV file
    df.to_csv(f'Analyses/User Data/usersUK_clustered_{event_name}_{period}.csv', index=False)

    # Display the resulting clusters
    print(df)

    # Step 3: Analyze and Visualize Results

    # Topic distribution
    topic_distribution = df['topic'].value_counts().sort_index()
    print(topic_distribution)

    # Optionally, display the topic distribution in a DataFrame for a better view
    topic_distribution_df = topic_distribution.reset_index()
    topic_distribution_df.columns = ['Topic', 'User Count']
    print(topic_distribution_df)

    # Calculate mean values of opinion columns for each cluster
    cluster_means = df.groupby('kmeans_cluster')[all_opinion_columns].mean()
    print(cluster_means)

    # Reduce dimensionality for visualization
    reducer = UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(df[all_opinion_columns])

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=df['kmeans_cluster'], cmap='Spectral', s=5)
    plt.colorbar(boundaries=np.arange(len(topic_distribution) + 1) - 0.5).set_ticks(np.arange(len(topic_distribution)))
    plt.title(f'UK User Clustering with BERTopic and KMeans ({event_name} - {period} period)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

# Perform clustering and visualization for different periods
for event_name in time_ranges.keys():
    df_before_clustered = perform_clustering_with_BERTopic(df_before, 'before', event_name)
    df_during_clustered = perform_clustering_with_BERTopic(df_during, 'during', event_name)
    df_after_clustered = perform_clustering_with_BERTopic(df_after, 'after', event_name)
