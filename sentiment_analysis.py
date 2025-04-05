from mpi4py import MPI
import json
from datetime import datetime
from collections import defaultdict
import heapq
import sys
import argparse
from math import ceil

def parse_datetime(dt_str):
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ")

def process_chunk(lines):
    """Process a chunk of JSON lines to compute sentiment per hour and per user."""
    if not lines:
        return {}, {}
    
    hour_sentiments = defaultdict(float)
    user_sentiments = defaultdict(float)

    for line in lines:
        try:
            if isinstance(line, str):
                data = json.loads(line)
            elif isinstance(line, dict):
                data = line
            else:
                continue
            
            doc = data.get('doc', {})
            if not isinstance(doc, dict):
                continue

            created_at = doc.get('createdAt')
            sentiment = doc.get('sentiment')
            account = doc.get('account', {})

            if not created_at or not isinstance(created_at, str):
                continue

            if sentiment is None or not isinstance(sentiment, (int, float)):
                continue

            if not isinstance(account, dict):
                continue

            user_id = account.get('id')
            username = account.get('username')

            if not (isinstance(user_id, str) and isinstance(username, str) and user_id.strip() and username.strip()):
                continue

            try:
                dt = parse_datetime(created_at)
            except Exception:
                continue

            hour_key = dt.strftime("%Y-%m-%d %H:00")
            user_key = (username, user_id)

            sentiment_float = float(sentiment)

            hour_sentiments[hour_key] += sentiment_float
            user_sentiments[user_key] += sentiment_float

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

    start_time = MPI.Wtime()
    
    if rank == 0:
        args = parser.parse_args()
        total_hour_sentiments = defaultdict(float)
        total_user_sentiments = defaultdict(float)
        
        try:
            with open(args.input_file, 'r') as f:
                while True:
                    lines = []
                    for _ in range(args.batch_size):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    
                    if not lines:
                        break
                    
                    if size > len(lines):
                        chunks = [[line] for line in lines] + [[] for _ in range(size - len(lines))]
                    else:
                        chunk_size = ceil(len(lines) / size)
                        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
                        if len(chunks) < size:
                            chunks += [[] for _ in range(size - len(chunks))]
                    
                    chunk = comm.scatter(chunks, root=0)
                    
                    local_hour_sentiments, local_user_sentiments = process_chunk(chunk)
                    
                    all_hour_sentiments = comm.gather(local_hour_sentiments, root=0)
                    all_user_sentiments = comm.gather(local_user_sentiments, root=0)
                    
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
        chunks = [None] * size
        comm.scatter(chunks, root=0)
        
        happiest_hours = heapq.nlargest(5, total_hour_sentiments.items(), key=lambda x: x[1])
        saddest_hours = heapq.nsmallest(5, total_hour_sentiments.items(), key=lambda x: x[1])
        happiest_users = heapq.nlargest(5, total_user_sentiments.items(), key=lambda x: x[1])
        saddest_users = heapq.nsmallest(5, total_user_sentiments.items(), key=lambda x: x[1])
        
        print("\n5 Happiest Hours:")
        for hour, sentiment in happiest_hours:
            print(f"Hour: {hour}, Sentiment Score: {sentiment:.8f}")
            
        print("\n5 Saddest Hours:")
        for hour, sentiment in saddest_hours:
            print(f"Hour: {hour}, Sentiment Score: {sentiment:.8f}")
            
        print("\n5 Happiest People:")
        for (username, user_id), sentiment in happiest_users:
            print(f"User: {username}, ID: {user_id}, Total Sentiment Score: {sentiment:.8f}")
            
        print("\n5 Saddest People:")
        for (username, user_id), sentiment in saddest_users:
            print(f"User: {username}, ID: {user_id}, Total Sentiment Score: {sentiment:.8f}")

        end_time = MPI.Wtime()
        print(f"\nTotal Execution Time: {end_time - start_time:.4f} seconds")
    
    else:
        while True:
            chunk = comm.scatter(None, root=0)

            if chunk is None:
                break

            local_hour_sentiments, local_user_sentiments = process_chunk(chunk)

            comm.gather(local_hour_sentiments, root=0)
            comm.gather(local_user_sentiments, root=0)

    # Synchronize all processes before exiting
    comm.Barrier()

if __name__ == "__main__":
    main()
    MPI.Finalize()
