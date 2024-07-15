import csv

def load_data(file_path):
    with open(file_path, newline='') as file:
        return list(csv.DictReader(file))

def validate_comments(data):
    id_lookup = {item['id']: item for item in data}
    to_delete = set()

    # Check if each comment's parent exists
    for item in data:
        if item['type'] == 'comment':
            # Ensure parent_id exists and is correctly formatted before processing
            if item['parent_id'] and len(item['parent_id']) > 3:
                parent_id = item['parent_id'][3:]  # Skipping the prefix 't1_' or 't3_'
                # Ensure the stripped parent_id exists in lookup or add to delete
                if parent_id not in id_lookup:
                    to_delete.add(item['id'])

    # Include descendants of invalid comments
    more_to_delete = True
    while more_to_delete:
        new_round = set()
        for item in data:
            if item['parent_id'] and len(item['parent_id']) > 3 and item['parent_id'][3:] in to_delete and item['id'] not in to_delete:
                new_round.add(item['id'])
        if not new_round:
            more_to_delete = False
        to_delete.update(new_round)

    return to_delete

def delete_records(data, to_delete):
    return [item for item in data if item['id'] not in to_delete]

def main(file_path):
    data = load_data(file_path)
    invalid_ids = validate_comments(data)
    print(f"Number of data points to be deleted: {len(invalid_ids)}")

    if invalid_ids:
        user_input = input("Do you want to rewrite the file without these data points? (yes/no): ")
        if user_input.lower() == 'yes':
            new_data = delete_records(data, invalid_ids)
            fieldnames = data[0].keys() if data else []
            with open(file_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(new_data)
            print("File has been rewritten.")
        else:
            print("No changes made to the file.")

if __name__ == '__main__':
    file_path = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/US/Subreddit Data/ClimateChangeUS_data.csv'
    main(file_path)
