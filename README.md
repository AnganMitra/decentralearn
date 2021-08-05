# Decentralized Online Personalized Learning [Privacy by design]

## High Level Control Parameters
1. Connectivity Graph of Neighbours 
2. Lookback interval
3. Lookahead period

## Storage Requirements 
1. Oracle 
2. Deep learning Model
3. Data buffer

## Features
1. Personalized Deep Learning Models with Identical Structure
2. Privacy by design since no node is aware of other's data.
3. Only model weights are exchange
4. Absence of a global model
5. Support Peer to Peer/Group Deep Learning

## Forecasting Experiments on Thailand's Smart Building Dataset
1. Isolated Learning (baseline model) vs Collective Learning with Neighbours
2. Effect of Network Topology on Learning  [cycle, grid, line, clique]
3. Preferential Learning amongst Partners

## Usage

usage: main.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR] [-sdt START_DATE] [-edt END_DATE] [-cdt CUT_DATE] [-fl FLOOR] [-zn NB_ZONE]
               [-grt GRAPH_TYPE] [-feat FEATURE] [-resm RESAMPLE_METHOD] [-model MODEL] [-plotFig PLOTFIG] [-modePred MODEPRED]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        path of the input data folder
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path of the output directory.
  -sdt START_DATE, --start_date START_DATE
                        start date: [YYYY-MM-DD]
  -edt END_DATE, --end_date END_DATE
                        end date: [YYYY-MM-DD]
  -cdt CUT_DATE, --cut_date CUT_DATE
                        cut date: [YYYY-MM-DD]
  -fl FLOOR, --floor FLOOR
                        floor number [int between 1 to 7 inclusive]
  -zn NB_ZONE, --nb_zone NB_ZONE
                        number of zones
  -grt GRAPH_TYPE, --graph_type GRAPH_TYPE
                        Types of graph: [grid, cycle, line, complete, isolated]
  -feat FEATURE, --feature FEATURE
                        Feature to learn : [temperature, humidity, power]
  -resm RESAMPLE_METHOD, --resample_method RESAMPLE_METHOD
                        Resample method: [min, max, sum]
  -model MODEL, --model MODEL
                        Choose between linear, seq2seq, lstm, cnn
  -plotFig PLOTFIG, --plotFig PLOTFIG
                        True to plot figures
  -modePred MODEPRED, --modePred MODEPRED
                        True to predict