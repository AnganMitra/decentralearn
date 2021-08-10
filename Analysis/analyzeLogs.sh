mkdir -p logs
mkdir -p plots

# floor wise comparision
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c floor -a 3,4,5,7

#graph wise comparison
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c graph -a grid,complete,line,cycle

#feature wise comparison
python3 analyzeLogs.py -i "./logs/" -o "./plots/" -c feature -a temperature 