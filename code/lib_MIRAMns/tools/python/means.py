import argparse
import json


parser = argparse.ArgumentParser(description='Small py file to compute means of execution time')

parser.add_argument('--input',  default="",
                    help='file to open (without extension)')
args = parser.parse_args()


# Open file
nb_edge = 0
path_input = args.input + ".log"
path_output = str(args.input) + ".means"
file  = open(path_input,"r",encoding="utf8")
output = open(path_output, "w")
lines = file.readlines()
dictio = {}
sumE = 0.0
nb_it = 0
for n, line in enumerate(lines) :
    list_word = line.split()
    if(len(list_word) > 0):
        sumE = sumE + float(list_word[0])
        nb_it = nb_it + 1

means = sumE / nb_it
output.write(str(means))
file.close()
output.close()