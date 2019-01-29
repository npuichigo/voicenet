#!/bin/bash

./local/run_tts.sh --input_dim=425 --output_dim=75 --batch_size=1 --learning_rate=0.001 --rnn_cell=fused_lstm
