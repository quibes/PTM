
import json
import os
from datetime import datetime

def filter_messages_by_date(directory_path, start_date, end_date, output_file, max_messages=30000):
    """
    Filters messages from JSON files in a directory based on a date range and saves the most recent messages to a file.

    Args:
        directory_path (str): Path to the directory containing JSON files.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        output_file (str): Name of the output file to save the filtered messages.
        max_messages (int): Maximum number of messages to include in the output file.
    """
    # Parse the date range
    include_start_date = datetime.strptime(start_date, "%Y-%m-%d")
    include_end_date = datetime.strptime(end_date, "%Y-%m-%d")

    filtered_messages = []

    # Iterate over all JSON files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(directory_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Load the JSON content
                    data = json.load(file)
                    messages = data.get("messages", [])

                    # Filter messages by date range
                    for message in messages:
                        message_date_str = message.get("date")
                        if not message_date_str:
                            continue

                        message_date = datetime.strptime(message_date_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=None)
                        if include_start_date <= message_date <= include_end_date:
                            filtered_messages.append({
                                "message": message,
                                "date": message_date
                            })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_name}: {e}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    # Sort the filtered messages by date in descending order
    filtered_messages.sort(key=lambda x: x["date"], reverse=True)

    # Select the latest messages up to the specified maximum
    latest_messages = [msg["message"] for msg in filtered_messages[:max_messages]]

    # Save the filtered messages to the output file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(latest_messages, out_file, ensure_ascii=False, indent=4)

    print(f"Filtered messages have been saved to {output_file}")
    return len(latest_messages)

if __name__ == "__main__":
    # Example usage
    directory_path = "data/json_files"
    start_date = "2023-10-01"
    end_date = "2024-11-15"
    output_file = "filtered_messages.json"
    max_messages = 30000

    message_count = filter_messages_by_date(directory_path, start_date, end_date, output_file, max_messages)
    print(f"Total messages saved: {message_count}")
