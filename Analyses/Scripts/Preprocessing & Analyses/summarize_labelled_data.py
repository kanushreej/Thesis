import pandas as pd

def summarize_labelled_data(issue):
    file_path = f'Analyses/Labelled Data/{issue}_labelled.csv'
    df = pd.read_csv(file_path)

    print("Column headers:")
    print(df.columns.tolist())
    
    print("\nTop 10 rows of the labelled data:\n")
    print(df.head(10))
    
    total_data_points = df.shape[0]
    print(f"\nTotal number of data points: {total_data_points}")
    
    stance_groups = {
        'Brexit': ['pro_brexit', 'anti_brexit'],
        'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUK': ['pro_NHS', 'anti_NHS'],
        'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
        'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation'],

        'ImmigrationUS': ['pro_immigration', 'anti_immigration'],
        'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUS': ['public_healthcare', 'private_healthcare'],
        'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
        'TaxationUS': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
    }
    
    stance_columns = stance_groups[issue] + ['neutral', 'irrelevant']
    
    print("\nCount for each stance:")
    for stance in stance_columns:
        stance_count = df[stance].sum()
        print(f"{stance}: {stance_count}")
    
    no_stance_rows = df[stance_columns].sum(axis=1) == 0
    no_stance_count = no_stance_rows.sum()
    if no_stance_count > 0:
        print(f"\nNumber of rows with no stance value: {no_stance_count}")
        print("Rows with no stance value:")
        print(df[no_stance_rows])
    else:
        print("\nNo rows with no stance value")

issue = 'Brexit'
summarize_labelled_data(issue)
