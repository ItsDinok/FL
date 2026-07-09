#!/bin/bash

python3 server.py & 
pid1=$!
python3 client.py --cid 0 &
pid2=$!
python3 client.py --cid 1 &
pid3=$!
python3 client.py --cid 2 &
pid4=$!

wait $pid1
echo "Script exited with code $?"
wait $pid2
echo "Script exited with code $?"
wait $pid3
echo "Script exited with code $?"
wait $pid4
echo "Script exited with code $?"

echo "All scripts completed."
