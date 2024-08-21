import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import norm
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Define major political events
events = {
    'Brexit Referendum': datetime(2016, 6, 23),
    #'US Presidential Election 2016': datetime(2016, 11, 8),
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
    'HealthcareUK': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/UK/HealthcareUK_labelled.csv',
    'IsraelPalestineUK': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/UK/IsraelPalestineUK_labelled.csv',
    'TaxationUK': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/UK/TaxationUK_labelled.csv'
}

#US
#topic_files = {
#    'ImmigrationUS': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/US/ImmigrationUS_labelled.csv',
#    'ClimateChangeUS': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/US/ClimateChangeUS_labelled.csv',
#    'HealthcareUS': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/US//HealthcareUS_labelled.csv',
#    'IsraelPalestineUS': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/US/IsraelPalestineUS_labelled.csv',
#    'TaxationUS': 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Analyses/Labelled Data/US/TaxationUS_labelled.csv'
#}

# Opinion columns for different topics
opinion_columns = {
    'Brexit': ['pro_brexit', 'anti_brexit'],
    'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
    'HealthcareUK': ['pro_NHS', 'anti_NHS'],
    'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
    'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation']
}

#US
#opinion_columns = {
#    'ImmigrationUS': ['pro_immigration', 'anti_immigration'],
#    'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
#    'HealthcareUS': ['public_healthcare', 'private_healthcare'],
#    'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
#    'TaxationUS': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
#}

# Generate time ranges for all events
time_ranges = {event: get_time_ranges(date) for event, date in events.items()}

def filter_and_cross_reference(event_name, time_ranges, topic_files, opinion_columns):
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

def aggregate_stances(time_ranges, topic_files, opinion_columns):
    df_before_list = []
    df_during_list = []
    df_after_list = []

    for event_name in time_ranges.keys():
        before, during, after = filter_and_cross_reference(event_name, time_ranges, topic_files, opinion_columns)
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

def perform_clustering_with_BERTopic(df, period, event_name):
    if df.empty:
        print(f"No data to cluster for {event_name} during {period} period")
        return df, None, None

    # Handle NaNs by filling with zero
    df = df.fillna(0)

    # Normalize the opinion columns
    scaler = StandardScaler()
    opinion_cols = [col for cols in opinion_columns.values() for col in cols]
    df[opinion_cols] = scaler.fit_transform(df[opinion_cols])

    # Filter users within 2 standard deviations
    df['total_opinions'] = df[opinion_cols].sum(axis=1)
    mu, std = norm.fit(df['total_opinions'])

    lower_bound = mu - 2 * std
    upper_bound = mu + 2 * std
    df = df[(df['total_opinions'] >= lower_bound) & (df['total_opinions'] <= upper_bound)]
    df.drop(columns=['total_opinions'], inplace=True, errors='ignore')

    # Perform clustering
    X = df[opinion_cols]
    cluster_model = KMeans(n_clusters=8)  
    clusters = cluster_model.fit_predict(X)
    df['cluster'] = clusters

    # Save the clustering results to a file
    df.to_csv(f'clustering_results_{event_name}_{period}.csv', index=False)

    # Calculate and save distance to cluster center for each user
    distances = cluster_model.transform(X)
    df['distance_to_center'] = [distances[i][clusters[i]] for i in range(len(clusters))]

    # Compute average distance to center
    avg_distance_to_center = df['distance_to_center'].mean()

    # Reduce dimensions for visualization
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean')
    umap_embeddings = umap_model.fit_transform(X)

    df['umap_x'] = umap_embeddings[:, 0]
    df['umap_y'] = umap_embeddings[:, 1]

    return df, cluster_model, avg_distance_to_center

def calculate_average_distances(df_before, df_during, df_after):
    avg_distance_before = df_before['distance_to_center'].mean() if 'distance_to_center' in df_before.columns else float('nan')
    avg_distance_during = df_during['distance_to_center'].mean() if 'distance_to_center' in df_during.columns else float('nan')
    avg_distance_after = df_after['distance_to_center'].mean() if 'distance_to_center' in df_after.columns else float('nan')
    return avg_distance_before, avg_distance_during, avg_distance_after

def plot_umap_clusters(df, title):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='umap_x', y='umap_y', hue='cluster', data=df, palette='tab10', marker='o')
    plt.title(title)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

def plot_cluster_distribution(df_before, df_during, df_after):
    cluster_counts_before = df_before['cluster'].value_counts().sort_index()
    cluster_counts_during = df_during['cluster'].value_counts().sort_index()
    cluster_counts_after = df_after['cluster'].value_counts().sort_index()

    cluster_labels = cluster_counts_before.index.tolist()
    cluster_counts_before = cluster_counts_before.reindex(cluster_labels, fill_value=0)
    cluster_counts_during = cluster_counts_during.reindex(cluster_labels, fill_value=0)
    cluster_counts_after = cluster_counts_after.reindex(cluster_labels, fill_value=0)

    df_distribution = pd.DataFrame({
        'Cluster': cluster_labels,
        'Before': cluster_counts_before,
        'During': cluster_counts_during,
        'After': cluster_counts_after
    })

    df_distribution.set_index('Cluster').plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab10')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Users')
    plt.title('Cluster Distribution Before, During, and After Event')
    plt.legend(title='Period')
    plt.grid(True)
    plt.show()

def plot_avg_distances(avg_distance_before, avg_distance_during, avg_distance_after, event_name):
    periods = ['Before', 'During', 'After']
    avg_distances = [avg_distance_before, avg_distance_during, avg_distance_after]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=periods, y=avg_distances, palette=['blue', 'orange', 'green'])
    plt.xlabel('Period')
    plt.ylabel('Average Distance to Cluster Center')
    plt.title(f'Average Distance to Cluster Center Before, During, and After {event_name}')
    plt.show()

def plot_polarization_over_time(avg_distance_before, avg_distance_during, avg_distance_after, event_name):
    periods = ['Before', 'During', 'After']
    avg_distances = [avg_distance_before, avg_distance_during, avg_distance_after]

    plt.figure(figsize=(10, 6))
    plt.plot(periods, avg_distances, marker='o', linestyle='-', color='b')
    plt.xlabel('Period')
    plt.ylabel('Average Distance to Cluster Center')
    plt.title(f'Polarization Over Time for {event_name}')
    plt.grid(True)
    plt.show()

def main():
    for event_name in events.keys():
        # Aggregating data
        df_before, df_during, df_after = aggregate_stances(time_ranges, topic_files, opinion_columns)

        # Print number of data points
        num_before = len(df_before)
        num_during = len(df_during)
        num_after = len(df_after)
        total_points = num_before + num_during + num_after

        print(f"Number of data points before {event_name}: {num_before}")
        print(f"Number of data points during {event_name}: {num_during}")
        print(f"Number of data points after {event_name}: {num_after}")
        print(f"Total number of data points for {event_name}: {total_points}")

        # Clustering
        df_before_clustered, cluster_model_before, avg_distance_before = perform_clustering_with_BERTopic(df_before, 'before', event_name)
        df_during_clustered, cluster_model_during, avg_distance_during = perform_clustering_with_BERTopic(df_during, 'during', event_name)
        df_after_clustered, cluster_model_after, avg_distance_after = perform_clustering_with_BERTopic(df_after, 'after', event_name)

        # Plot UMAP visualizations
        if not df_before_clustered.empty:
            plot_umap_clusters(df_before_clustered, f'UMAP Visualization Before {event_name}')
        if not df_during_clustered.empty:
            plot_umap_clusters(df_during_clustered, f'UMAP Visualization During {event_name}')
        if not df_after_clustered.empty:
            plot_umap_clusters(df_after_clustered, f'UMAP Visualization After {event_name}')

        # Plot Cluster Distribution
        if not df_before_clustered.empty and not df_during_clustered.empty and not df_after_clustered.empty:
            plot_cluster_distribution(df_before_clustered, df_during_clustered, df_after_clustered)

        # Calculate average distances
        avg_distance_before, avg_distance_during, avg_distance_after = calculate_average_distances(df_before_clustered, df_during_clustered, df_after_clustered)
        
        print(f"Average distance to cluster center before {event_name}: {avg_distance_before}")
        print(f"Average distance to cluster center during {event_name}: {avg_distance_during}")
        print(f"Average distance to cluster center after {event_name}: {avg_distance_after}")

        # Plot average distances
        plot_avg_distances(avg_distance_before, avg_distance_during, avg_distance_after, event_name)
        
        # Plot polarization over time
        plot_polarization_over_time(avg_distance_before, avg_distance_during, avg_distance_after, event_name)

if __name__ == "__main__":
    main()
