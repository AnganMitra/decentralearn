# floor wise comparision
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c floor 

#graph wise comparison
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c grid
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c complete
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c line
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c cycle 

#feature wise comparison
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c temperature 