from mpi4py import MPI
import json
from datetime import datetime
from collections import defaultdict
import heapq
import sys
import argparse

def parse_datetime(dt_str):
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ")

def process_chunk(lines):
    """Process a chunk of JSON lines to compute sentiment per hour and per user."""
    hour_sentiments = defaultdict(float)
    user_sentiments = defaultdict(float)
    
    
    
    for line in lines:
        try:
            # Ensure line is a string before parsing
            if isinstance(line, str):
                data = json.loads(line)
            elif isinstance(line, dict):  # If already parsed, use it directly
                data = line
            else:
                continue  # Skip invalid data
            

            doc = data.get('doc', {})
            if not isinstance(doc, dict):
                continue  # Skip this line if 'doc' is missing or invalid
            
            # Extract required fields
            created_at = doc.get('createdAt')
            sentiment = doc.get('sentiment')
            account = doc.get('account', {})

            # Check if required fields are present and valid
            if not created_at or not isinstance(created_at, str):
                continue  # Skip if 'createdAt' is missing or not a string

            if sentiment is None or not isinstance(sentiment, (int, float)):
                continue  # Skip if 'sentiment' is missing or not a number

            if not isinstance(account, dict):
                continue  # Skip if 'account' is not a dictionary

            user_id = account.get('id')
            username = account.get('username')

            # Validate user_id and username
            if not (isinstance(user_id, str) and isinstance(username, str) and user_id.strip() and username.strip()):
                continue

            # Parse datetime safely
            try:
                dt = parse_datetime(created_at)
            except Exception as e:
                continue  # Skip line if the datetime format is invalid

            # Process hour sentiment
            hour_key = dt.strftime("%Y-%m-%d %H:00")
            hour_sentiments[hour_key] += float(sentiment)

            # Process user sentiment
            user_key = (username, user_id)
            user_sentiments[user_key] += float(sentiment)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Skipping malformed entry: {e}", file=sys.stderr)
            continue
    
    return hour_sentiments, user_sentiments

def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment from Mastodon data')
    parser.add_argument('input_file', help='Path to the input NDJSON file')
    parser.add_argument('--batch-size', type=int, default=5000, 
                      help='Number of lines to process in each batch (default: 5000)')
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start timing
    start_time = MPI.Wtime()
    
    if rank == 0:
        args = parser.parse_args()
        total_hour_sentiments = defaultdict(float)
        total_user_sentiments = defaultdict(float)
        
        try:
            with open(args.input_file, 'r') as f:
                while True:
                    # Read batch_size lines
                    lines = []
                    for _ in range(args.batch_size):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    
                    if not lines:
                        break
                        
                    # Calculate chunk size and distribute data for this batch
                    chunk_size = len(lines) // size
                    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
                    
                    # If there are remaining lines, add them to the last chunk
                    if len(chunks) > size:
                        chunks[size-1].extend(chunks[size:])
                        chunks = chunks[:size]
                        
                    # Broadcast batch to all processes
                    chunk = comm.scatter(chunks, root=0)
                    
                    # Process local chunk
                    local_hour_sentiments, local_user_sentiments = process_chunk(chunk)
                    
                    # Gather results from all processes
                    all_hour_sentiments = comm.gather(local_hour_sentiments, root=0)
                    all_user_sentiments = comm.gather(local_user_sentiments, root=0)
                    
                    # Combine batch results
                    for hour_sent in all_hour_sentiments:
                        for hour, sentiment in hour_sent.items():
                            total_hour_sentiments[hour] += sentiment
                            
                    for user_sent in all_user_sentiments:
                        for user, sentiment in user_sent.items():
                            total_user_sentiments[user] += sentiment
                    
                    print(f"Processed {len(lines)} lines...", file=sys.stderr)
                    
        except FileNotFoundError:
            print(f"Error: File '{args.input_file}' not found")
            comm.Abort(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            comm.Abort(1)

        # Send termination signal (None) to all workers
        for _ in range(size - 1):
            comm.scatter(None, root=0)
            
        # Get final results
        happiest_hours = heapq.nlargest(5, total_hour_sentiments.items(), key=lambda x: x[1])
        saddest_hours = heapq.nsmallest(5, total_hour_sentiments.items(), key=lambda x: x[1])
        happiest_users = heapq.nlargest(5, total_user_sentiments.items(), key=lambda x: x[1])
        saddest_users = heapq.nsmallest(5, total_user_sentiments.items(), key=lambda x: x[1])
        
        # Print results
        print("\n5 Happiest Hours:")
        for hour, sentiment in happiest_hours:
            print(f"Hour: {hour}, Sentiment Score: {sentiment:.2f}")
            
        print("\n5 Saddest Hours:")
        for hour, sentiment in saddest_hours:
            print(f"Hour: {hour}, Sentiment Score: {sentiment:.2f}")
            
        print("\n5 Happiest People:")
        for (username, user_id), sentiment in happiest_users:
            print(f"User: {username}, ID: {user_id}, Total Sentiment Score: {sentiment:.2f}")
            
        print("\n5 Saddest People:")
        for (username, user_id), sentiment in saddest_users:
            print(f"User: {username}, ID: {user_id}, Total Sentiment Score: {sentiment:.2f}")

        # End timing
        end_time = MPI.Wtime()
        print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    
    else:
        while True:
            # Receive chunks from root
            chunk = comm.scatter(None, root=0)

            # Termination condition: if root sends None, exit loop
            if chunk is None:
                break

            # Process the chunk
            local_hour_sentiments, local_user_sentiments = process_chunk(chunk)

            # Send results back to root
            comm.gather(local_hour_sentiments, root=0)
            comm.gather(local_user_sentiments, root=0)

if __name__ == "__main__":
    main()